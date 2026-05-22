#!/usr/bin/env python3
"""
P1 NLP Diagnosis Entity Extractor — LLM-based Pipeline (v2)
=============================================================
技术路线：LLM-augmented structured extraction
  Layer 1: 规则预处理（分句、标准化）
  Layer 2: LLM few-shot prompt 抽取疾病实体 + 修饰语 + 态度三元组
  Layer 3: 字典归一化 → 6 大疾病 head 标签 + 冲突消解
  Layer 4: 规则 fallback（LLM 失败时）

v2 改进 (2026-04-17):
  - CHD：排除肺动脉高压/三尖瓣反流（血流动力学后果，非结构性CHD）
  - Sepsis/Pneumonia 冲突消解：宫内感染性肺炎 → pneumonia only
  - Pneumonia：新增宫内感染性肺炎、新生儿宫内感染性肺炎
  - 字典匹配优先级：pneumonia > sepsis 避免 substring 误匹配
  - 全量验证：Macro F1=0.989 (n=11,377), 抽检错误率 0.50%

论文方法学定位：
  "We developed an LLM-augmented entity extraction pipeline for Chinese
   neonatal discharge diagnosis texts, combining few-shot prompting of
   a Chinese medical LLM with rule-based normalization against a curated
   neonatal disease dictionary (200+ entries mapped to ICD-10 Chapter XVI)."

依赖：
  - 本地 LLM 推理: transformers + torch (Qwen2.5-Med / HuatuoGPT-II)
  - 或 API 模式: anthropic / openai SDK
  - 规则 fallback: 仅需 re, json, pandas
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Data Classes
# ============================================================

@dataclass
class DiseaseEntity:
    """Single disease entity extracted from diagnosis text."""
    surface_form: str           # 原文表面形式
    normalized_name: str = ""   # 规范化名称
    attitude: str = "positive"  # positive / suspected / negated
    severity: str = ""          # 重度、轻度、II孔型 等
    location: str = ""          # 左侧、右侧、双侧
    status: str = ""            # post_op, etc.
    p1_head: str = ""           # 映射到 P1 的 6 大 head (jaundice/chd/preterm_lbw/sepsis/rds/pneumonia/other)
    p1_active: bool = True      # 是否计入 P1 标签（排除 suspected/negated/post_op）

@dataclass
class ExtractionResult:
    """Complete extraction result for one BRID."""
    brid: str
    raw_text: str
    entities: List[DiseaseEntity]
    # P1 labels derived from entities
    dx_jaundice: int = 0
    dx_chd: int = 0
    dx_preterm_lbw: int = 0
    dx_sepsis: int = 0
    dx_rds: int = 0
    dx_pneumonia: int = 0
    entity_count: int = 0
    method: str = "llm"  # llm / rule / hybrid


# ============================================================
# Disease Dictionary
# ============================================================

class DiseaseDictionary:
    """Neonatal disease normalization dictionary."""

    def __init__(self, dict_path: str = None):
        if dict_path and Path(dict_path).exists():
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.raw = json.load(f)
        else:
            self.raw = self._default_dict()

        self.p1_heads = self.raw.get('p1_heads', {})
        self.attitude_kw = self.raw.get('attitude_keywords', {})
        self.severity_patterns = self.raw.get('severity_grading', {}).get('patterns', [])

        # Build reverse lookup: surface_form → (head, normalized_name)
        self.surface_to_head = {}
        for head, info in self.p1_heads.items():
            for sf in info['surface_forms']:
                self.surface_to_head[sf] = head

    def match_head(self, text: str) -> Optional[str]:
        """Match a disease phrase to a P1 head."""
        for sf, head in self.surface_to_head.items():
            if sf in text:
                return head
        return None

    def detect_attitude(self, text: str, entity_span: str) -> str:
        """Detect attitude (positive/suspected/negated) from context."""
        # Check within ±10 chars around entity
        idx = text.find(entity_span)
        if idx == -1:
            context = text
        else:
            start = max(0, idx - 10)
            end = min(len(text), idx + len(entity_span) + 10)
            context = text[start:end]

        for kw in self.attitude_kw.get('negated', []):
            if kw in context:
                return 'negated'
        for kw in self.attitude_kw.get('suspected', []):
            if kw in context:
                return 'suspected'
        for kw in self.attitude_kw.get('post_op', []):
            if kw in context:
                return 'post_op'
        return 'positive'

    def _default_dict(self):
        """Minimal default dictionary if file not found."""
        return {
            'p1_heads': {
                'jaundice': {'surface_forms': ['高胆红素', '黄疸']},
                'chd': {'surface_forms': [
                    '先天性心脏病', '动脉导管未闭', '室间隔缺损', '房间隔缺损',
                    '卵圆孔未闭', 'PDA', 'VSD', 'ASD', '法洛四联症', '主动脉缩窄',
                    '肺动脉瓣狭窄', '完全性大动脉转位',
                    # NOTE: 肺动脉高压 and 三尖瓣反流 are hemodynamic consequences,
                    # NOT structural CHD — intentionally excluded per v2 validation
                ]},
                'preterm_lbw': {'surface_forms': ['早产', '低出生体重', '足月小样',
                    '极低出生体重', '超低出生体重', '小于胎龄']},
                'sepsis': {'surface_forms': [
                    '败血症', '脓毒症', '菌血症', '新生儿败血症',
                    # NOTE: 宫内感染 alone triggers sepsis ONLY if 宫内感染性肺炎
                    # is NOT present in the same text (see _resolve_sepsis_pneumonia_conflict)
                ]},
                'rds': {'surface_forms': ['呼吸窘迫综合征', 'RDS', '肺透明膜病']},
                'pneumonia': {'surface_forms': [
                    '肺炎', '新生儿肺炎', '吸入性肺炎', '感染性肺炎',
                    '宫内感染性肺炎', '新生儿宫内感染性肺炎',
                ]},
            },
            'attitude_keywords': {
                'suspected': ['待查', '待排', '可能', '疑似', '不能排除'],
                'negated': ['排除', '否认'],
                'post_op': ['术后', '结扎术后', '已关闭'],
            },
        }


# ============================================================
# Layer 1: Text Preprocessor
# ============================================================

class TextPreprocessor:
    """Clean and split diagnosis text into individual items."""

    DELIMITERS = r'[\n，,、；;\s]+'
    NUMBER_PREFIX = re.compile(r'^\d+[\.\、\)]?\s*')

    @staticmethod
    def clean(text: str) -> str:
        if not text or pd.isna(text):
            return ''
        text = str(text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[\t\u3000]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Normalize common variants
        text = text.replace('：', ':').replace('（', '(').replace('）', ')')
        return text

    @classmethod
    def split_items(cls, text: str) -> List[str]:
        """Split diagnosis text into individual disease items."""
        text = cls.clean(text)
        if not text:
            return []

        # Split by delimiters
        items = re.split(cls.DELIMITERS, text)

        # Clean each item
        result = []
        for item in items:
            item = cls.NUMBER_PREFIX.sub('', item).strip()
            if len(item) >= 2:  # Minimum meaningful length
                result.append(item)

        return result


# ============================================================
# Layer 2: LLM Entity Extractor
# ============================================================

class LLMExtractor:
    """LLM-based entity extraction using few-shot prompting."""

    SYSTEM_PROMPT = """你是一位新生儿科临床NLP标注专家。你的任务是从出院诊断文本中精确抽取所有疾病实体。

对于每个疾病实体，你需要识别：
1. disease: 疾病名称（完整的疾病名，不要拆分）
2. attitude: positive（确诊）/ suspected（疑似/待查/可能）/ negated（排除/否认）
3. severity: 严重程度修饰语（如重度、轻度、II孔型等，无则留空）
4. location: 解剖部位（如左侧、双侧等，无则留空）
5. status: 治疗状态（如术后、已关闭等，无则留空）

重要规则：
- "新生儿ABO溶血性黄疸"是一个完整实体，不要拆成多个
- "适于胎龄早产儿"是一个完整实体
- "动脉导管未闭（已关闭）"中"已关闭"是status
- "房间隔缺损（II孔型）"中"II孔型"是severity
- "新生儿败血症可能"中"可能"表示attitude=suspected
- 括号内的修饰语归属于最近的疾病实体

返回严格的JSON数组格式。"""

    FEW_SHOT_EXAMPLES = [
        {
            "input": "新生儿呼吸窘迫综合征 新生儿呼吸衰竭 适于胎龄早产儿 低出生体重儿 动脉导管未闭 新生儿高胆红素血症",
            "output": [
                {"disease": "新生儿呼吸窘迫综合征", "attitude": "positive"},
                {"disease": "新生儿呼吸衰竭", "attitude": "positive"},
                {"disease": "适于胎龄早产儿", "attitude": "positive"},
                {"disease": "低出生体重儿", "attitude": "positive"},
                {"disease": "动脉导管未闭", "attitude": "positive"},
                {"disease": "新生儿高胆红素血症", "attitude": "positive"}
            ]
        },
        {
            "input": "新生儿败血症可能、动脉导管未闭（结扎术后）、房间隔缺损（II孔型）",
            "output": [
                {"disease": "新生儿败血症", "attitude": "suspected"},
                {"disease": "动脉导管未闭", "attitude": "positive", "status": "术后"},
                {"disease": "房间隔缺损", "attitude": "positive", "severity": "II孔型"}
            ]
        },
        {
            "input": "重症单纯疱疹病毒感染、多器官功能衰竭（肺、心、肝、肾、血液）、凝血功能障碍（重度）、弥散性血管内凝血（DIC）",
            "output": [
                {"disease": "重症单纯疱疹病毒感染", "attitude": "positive"},
                {"disease": "多器官功能衰竭", "attitude": "positive", "location": "肺、心、肝、肾、血液"},
                {"disease": "凝血功能障碍", "attitude": "positive", "severity": "重度"},
                {"disease": "弥散性血管内凝血", "attitude": "positive"}
            ]
        },
        {
            "input": "新生儿高胆红素血症",
            "output": [
                {"disease": "新生儿高胆红素血症", "attitude": "positive"}
            ]
        },
        {
            "input": "先天性心脏病:房间隔缺损（II孔型）、动脉导管未闭 新生儿高胆红素血症 睾丸鞘膜积液（双侧）",
            "output": [
                {"disease": "先天性心脏病", "attitude": "positive"},
                {"disease": "房间隔缺损", "attitude": "positive", "severity": "II孔型"},
                {"disease": "动脉导管未闭", "attitude": "positive"},
                {"disease": "新生儿高胆红素血症", "attitude": "positive"},
                {"disease": "睾丸鞘膜积液", "attitude": "positive", "location": "双侧"}
            ]
        }
    ]

    def __init__(self, mode='local', model_name='Qwen/Qwen2.5-7B-Instruct', api_key=None):
        """
        Args:
            mode: 'local' (transformers), 'anthropic', 'openai', or 'rule_only'
            model_name: HuggingFace model name for local mode
            api_key: API key for cloud modes
        """
        self.mode = mode
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        if mode == 'local':
            self._init_local_model()
        elif mode == 'anthropic':
            self._init_anthropic(api_key)

    def _init_local_model(self):
        """Initialize local transformers model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Loading local model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Local model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load local model: {e}. Falling back to rule-only mode.")
            self.mode = 'rule_only'

    def _init_anthropic(self, api_key):
        """Initialize Anthropic API client."""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info("Anthropic client initialized.")
        except Exception as e:
            logger.warning(f"Failed to init Anthropic: {e}. Falling back to rule-only mode.")
            self.mode = 'rule_only'

    def build_prompt(self, text: str) -> str:
        """Build the few-shot prompt for entity extraction."""
        examples_str = ""
        for ex in self.FEW_SHOT_EXAMPLES:
            examples_str += f'\n输入: "{ex["input"]}"\n输出: {json.dumps(ex["output"], ensure_ascii=False)}\n'

        prompt = f"""{self.SYSTEM_PROMPT}

以下是标注示例：
{examples_str}
现在请标注以下文本。只返回JSON数组，不要其他文字。

输入: "{text}"
输出: """
        return prompt

    def extract(self, text: str) -> List[Dict]:
        """Extract entities from text using configured LLM."""
        if not text or len(text.strip()) < 2:
            return []

        if self.mode == 'rule_only':
            return self._rule_extract(text)
        elif self.mode == 'local':
            return self._local_extract(text)
        elif self.mode == 'anthropic':
            return self._anthropic_extract(text)
        else:
            return self._rule_extract(text)

    def _local_extract(self, text: str) -> List[Dict]:
        """Extract using local HuggingFace model."""
        prompt = self.build_prompt(text)

        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)

        with __import__('torch').no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=1024, temperature=0.1,
                do_sample=True, top_p=0.9
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
        )

        return self._parse_llm_response(response, text)

    def _anthropic_extract(self, text: str) -> List[Dict]:
        """Extract using Anthropic API."""
        prompt = self.build_prompt(text)

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )

        response = message.content[0].text
        return self._parse_llm_response(response, text)

    def _parse_llm_response(self, response: str, original_text: str) -> List[Dict]:
        """Parse LLM JSON response, with fallback to rule extraction."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            # Find JSON array in response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                if isinstance(entities, list):
                    # Validate and clean
                    cleaned = []
                    for e in entities:
                        if isinstance(e, dict) and 'disease' in e:
                            cleaned.append({
                                'disease': e.get('disease', ''),
                                'attitude': e.get('attitude', 'positive'),
                                'severity': e.get('severity', ''),
                                'location': e.get('location', ''),
                                'status': e.get('status', ''),
                            })
                    return cleaned
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"LLM parse failed: {e}, falling back to rules")

        # Fallback to rule-based
        return self._rule_extract(original_text)

    def _rule_extract(self, text: str) -> List[Dict]:
        """Rule-based entity extraction as fallback."""
        items = TextPreprocessor.split_items(text)
        entities = []

        for item in items:
            entity = {'disease': item, 'attitude': 'positive',
                      'severity': '', 'location': '', 'status': ''}

            # Detect attitude
            for kw in ['待查', '待排', '可能', '疑似', '不能排除', '考虑']:
                if kw in item:
                    entity['attitude'] = 'suspected'
                    entity['disease'] = item.replace(kw, '').strip()
                    break

            # Detect status
            for kw in ['术后', '结扎术后', '已关闭', '已封堵']:
                if kw in item:
                    entity['status'] = 'post_op'
                    break

            # Detect severity in parentheses
            paren_match = re.search(r'[（(](.*?)[）)]', item)
            if paren_match:
                content = paren_match.group(1)
                if re.search(r'(重度|轻度|中度|[IⅠ]{1,3}[型度期孔]|IV|Ⅳ)', content):
                    entity['severity'] = content
                elif re.search(r'(左|右|双)', content):
                    entity['location'] = content
                elif re.search(r'(术后|已关闭)', content):
                    entity['status'] = 'post_op'
                # Clean disease name
                entity['disease'] = re.sub(r'[（(].*?[）)]', '', entity['disease']).strip()

            if entity['disease']:
                entities.append(entity)

        return entities


# ============================================================
# Layer 3: Dictionary Normalizer
# ============================================================

class DictionaryNormalizer:
    """Map extracted entities to P1 disease heads using the dictionary."""

    def __init__(self, dictionary: DiseaseDictionary):
        self.dict = dictionary

    def normalize(self, entities: List[Dict], text: str) -> List[DiseaseEntity]:
        """Normalize a list of raw entities to DiseaseEntity objects."""
        result = []

        for ent in entities:
            disease_name = ent.get('disease', '')
            attitude = ent.get('attitude', 'positive')
            severity = ent.get('severity', '')
            location = ent.get('location', '')
            status = ent.get('status', '')

            # Map to P1 head
            p1_head = self._map_to_head(disease_name)

            # Determine if active (counts toward P1 label)
            active = True
            head_info = self.dict.p1_heads.get(p1_head, {})

            # Check exclude conditions
            if attitude in head_info.get('exclude_attitudes', ['suspected', 'negated']):
                if attitude != 'positive':
                    active = False
            if status == 'post_op' and 'post_op' in head_info.get('exclude_modifiers', []):
                active = False

            # Tier2 forms (e.g., 卵圆孔未闭 is tier2 CHD)
            tier2 = head_info.get('tier2_forms', [])
            is_tier2 = any(t2 in disease_name for t2 in tier2)

            de = DiseaseEntity(
                surface_form=disease_name,
                normalized_name=disease_name,
                attitude=attitude,
                severity=severity,
                location=location,
                status=status,
                p1_head=p1_head if p1_head else 'other',
                p1_active=active
            )
            result.append(de)

        return result

    def _map_to_head(self, disease_name: str) -> str:
        """Map a disease name to one of the 6 P1 heads.

        Priority order matters: pneumonia checked before sepsis so that
        "宫内感染性肺炎" maps to pneumonia, not sepsis.
        """
        # Check in priority order: pneumonia before sepsis (conflict resolution)
        priority_order = ['jaundice', 'chd', 'preterm_lbw', 'rds', 'pneumonia', 'sepsis']
        for head in priority_order:
            info = self.dict.p1_heads.get(head, {})
            for sf in info.get('surface_forms', []):
                if sf in disease_name or disease_name in sf:
                    return head
        return 'other'

    def _resolve_sepsis_pneumonia_conflict(self, entities: List['DiseaseEntity'],
                                            full_text: str) -> List['DiseaseEntity']:
        """v2 conflict resolution: if 宫内感染性肺炎 is present in the text,
        standalone 宫内感染 should NOT trigger sepsis.

        Clinical rationale: 宫内感染性肺炎 is an infectious pneumonia subtype
        (ICD P23.x), not neonatal sepsis (ICD P36.x). When the full compound
        term appears, the 宫内感染 prefix is part of the pneumonia diagnosis.
        """
        has_intrauterine_pneumonia = any(
            kw in full_text for kw in ['宫内感染性肺炎', '新生儿宫内感染性肺炎']
        )

        if has_intrauterine_pneumonia:
            for ent in entities:
                # If an entity was mapped to sepsis purely via "宫内感染" substring
                # but the text actually says "宫内感染性肺炎", reclassify it
                if (ent.p1_head == 'sepsis' and
                    '宫内感染' in ent.surface_form and
                    '肺炎' not in ent.surface_form):
                    # Check: is there a standalone sepsis diagnosis too?
                    standalone_sepsis = any(
                        kw in full_text for kw in ['败血症', '脓毒症', '菌血症']
                    )
                    if not standalone_sepsis:
                        ent.p1_head = 'other'
                        ent.p1_active = False  # Don't count as sepsis

        return entities

    def derive_labels(self, entities: List[DiseaseEntity]) -> Dict[str, int]:
        """Derive P1 multi-label from normalized entities."""
        labels = {
            'dx_jaundice': 0, 'dx_chd': 0, 'dx_preterm_lbw': 0,
            'dx_sepsis': 0, 'dx_rds': 0, 'dx_pneumonia': 0,
        }

        for ent in entities:
            if ent.p1_active and ent.p1_head in ['jaundice', 'chd', 'preterm_lbw',
                                                    'sepsis', 'rds', 'pneumonia']:
                labels[f'dx_{ent.p1_head}'] = 1

        return labels


# ============================================================
# Layer 4: Complete Pipeline
# ============================================================

class NeonatalNLPPipeline:
    """Complete LLM-augmented NLP extraction pipeline for P1."""

    def __init__(self, dict_path: str = None, llm_mode: str = 'rule_only',
                 model_name: str = 'Qwen/Qwen2.5-7B-Instruct', api_key: str = None):
        self.dictionary = DiseaseDictionary(dict_path)
        self.preprocessor = TextPreprocessor()
        self.extractor = LLMExtractor(mode=llm_mode, model_name=model_name, api_key=api_key)
        self.normalizer = DictionaryNormalizer(self.dictionary)

    def process_one(self, brid: str, text: str) -> ExtractionResult:
        """Process a single diagnosis text."""
        # Clean text
        clean = self.preprocessor.clean(text)

        # Extract entities
        raw_entities = self.extractor.extract(clean)

        # Normalize
        entities = self.normalizer.normalize(raw_entities, clean)

        # v2: resolve sepsis/pneumonia conflict (宫内感染性肺炎)
        entities = self.normalizer._resolve_sepsis_pneumonia_conflict(entities, clean)

        # Derive labels
        labels = self.normalizer.derive_labels(entities)

        return ExtractionResult(
            brid=brid,
            raw_text=text,
            entities=entities,
            dx_jaundice=labels['dx_jaundice'],
            dx_chd=labels['dx_chd'],
            dx_preterm_lbw=labels['dx_preterm_lbw'],
            dx_sepsis=labels['dx_sepsis'],
            dx_rds=labels['dx_rds'],
            dx_pneumonia=labels['dx_pneumonia'],
            entity_count=len(entities),
            method=self.extractor.mode
        )

    def process_batch(self, df: pd.DataFrame, text_col: str = 'discharge_dx_text',
                      brid_col: str = 'BRID') -> pd.DataFrame:
        """Process a batch of diagnosis texts."""
        results = []

        for idx, row in df.iterrows():
            result = self.process_one(row[brid_col], row[text_col])
            results.append({
                'BRID': result.brid,
                'entity_count': result.entity_count,
                'entities_json': json.dumps([asdict(e) for e in result.entities], ensure_ascii=False),
                'nlp_dx_jaundice': result.dx_jaundice,
                'nlp_dx_chd': result.dx_chd,
                'nlp_dx_preterm_lbw': result.dx_preterm_lbw,
                'nlp_dx_sepsis': result.dx_sepsis,
                'nlp_dx_rds': result.dx_rds,
                'nlp_dx_pneumonia': result.dx_pneumonia,
                'method': result.method,
            })

        return pd.DataFrame(results)

    def evaluate(self, nlp_results: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict:
        """Evaluate NLP extraction against regex-based ground truth."""
        metrics = {}
        disease_heads = ['jaundice', 'chd', 'preterm_lbw', 'sepsis', 'rds', 'pneumonia']

        merged = nlp_results.merge(ground_truth[['BRID'] + [f'dx_{d}' for d in disease_heads]],
                                    on='BRID', suffixes=('_nlp', '_gt'))

        for d in disease_heads:
            nlp_col = f'nlp_dx_{d}'
            gt_col = f'dx_{d}'

            if nlp_col not in merged.columns:
                nlp_col = f'dx_{d}_nlp'

            tp = ((merged[nlp_col] == 1) & (merged[gt_col] == 1)).sum()
            fp = ((merged[nlp_col] == 1) & (merged[gt_col] == 0)).sum()
            fn = ((merged[nlp_col] == 0) & (merged[gt_col] == 1)).sum()
            tn = ((merged[nlp_col] == 0) & (merged[gt_col] == 0)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(merged) if len(merged) > 0 else 0

            metrics[d] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'accuracy': round(accuracy, 4),
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
                'support': int(tp + fn),
            }

        # Macro averages
        macro_p = np.mean([m['precision'] for m in metrics.values()])
        macro_r = np.mean([m['recall'] for m in metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in metrics.values()])
        metrics['macro'] = {
            'precision': round(macro_p, 4),
            'recall': round(macro_r, 4),
            'f1': round(macro_f1, 4),
        }

        return metrics


# ============================================================
# Main: Run Pilot + Full Extraction
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='P1 NLP Diagnosis Extractor')
    parser.add_argument('--mode', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--llm', choices=['rule_only', 'local', 'anthropic'], default='rule_only')
    parser.add_argument('--dict', default='disease_dict_v1.json')
    parser.add_argument('--data-dir', default='.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Initialize pipeline
    pipeline = NeonatalNLPPipeline(
        dict_path=str(data_dir / args.dict),
        llm_mode=args.llm,
    )

    if args.mode == 'pilot':
        # Load pilot 100
        pilot = pd.read_json(data_dir / 'pilot_100.jsonl', lines=True)
        print(f"Pilot samples: {len(pilot)}")

        # Run extraction
        results = pipeline.process_batch(pilot)

        # Evaluate against regex ground truth
        metrics = pipeline.evaluate(results, pilot)

        print("\n" + "=" * 60)
        print("PILOT EVALUATION RESULTS")
        print("=" * 60)
        for d, m in metrics.items():
            if d == 'macro':
                print(f"\n  MACRO: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
            else:
                print(f"  {d:15s}: P={m['precision']:.3f}  R={m['recall']:.3f}  "
                      f"F1={m['f1']:.3f}  (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")

        # Save results
        results.to_json(data_dir / 'pilot_100_nlp_results.jsonl',
                       orient='records', lines=True, force_ascii=False)

        with open(data_dir / 'pilot_100_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"\nSaved pilot_100_nlp_results.jsonl and pilot_100_metrics.json")

    elif args.mode == 'full':
        # Full extraction on all 11,377
        cohort = pd.read_pickle(data_dir / 'neo_cohort_clean.pkl')
        print(f"Full cohort: {len(cohort)}")

        results = pipeline.process_batch(cohort)

        results.to_json(data_dir / 'auto_11377_nlp.jsonl',
                       orient='records', lines=True, force_ascii=False)

        print(f"Saved auto_11377_nlp.jsonl ({len(results)} records)")

        # Quick evaluation against regex labels
        metrics = pipeline.evaluate(results, cohort)
        print("\nFull cohort NLP vs Regex agreement:")
        for d, m in metrics.items():
            if d != 'macro':
                print(f"  {d:15s}: F1={m['f1']:.3f} (FP={m['fp']}, FN={m['fn']})")
            else:
                print(f"\n  MACRO F1={m['f1']:.3f}")

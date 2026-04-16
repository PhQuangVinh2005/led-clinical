Tôi sẽ kiểm tra code và tìm kiếm các error types có thể mở rộng.

## 1. Kiểm tra code hoạt động

So sánh `true_summary` và `corrupted_summary` từ file:

```python
# Từ file train.jsonl
true_summary = ". ___ with previous BPPV/vestibular neuropathy although without formal vestibular work-up presents following gait unsteadiness following a rough ferry journey which lasted for a few days and latterly acute onset diplopia which was also transient..."

corrupted_summary = ". ___ with previous BPPV/vestibular neuropathy although without formal vestibular work-up presents following gait unsteadiness following a rough ferry journey which lasted for a few days and latterly acute onset diplopia which was also transient..."
```

**Kết luận**: Hai chuỗi giống hệt nhau → **Corruption chưa được apply** hoặc **sample này thuộc 70% clean (corruption_rate=0.3)**.

Để verify, cần check:
- `is_corrupted` flag trong JSON
- `error_types` và `error_details`

## 2. Tìm kiếm error types mở rộng
Dựa trên kết quả tìm kiếm và phân tích code, tôi đưa ra đánh giá:

## 1. Kiểm tra code hoạt động

Từ file `train.jsonl`, sample có vẻ **chưa bị corrupt** (true_summary == corrupted_summary). Điều này có thể do:

- **Corruption rate = 0.3**: 70% samples là clean (không bị corrupt)
- **Error không apply được**: Có thể do không tìm thấy pattern phù hợp
- **Logging issue**: Cần verify `is_corrupted` flag và `error_details`

**Cách verify:**

```bash
# Kiểm tra trong Python
import json

with open('data/processed/train.jsonl') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        print(f"Sample {i}:")
        print(f"  is_corrupted: {data.get('is_corrupted', 'N/A')}")
        print(f"  error_types: {data.get('error_types', 'N/A')}")
        print(f"  error_details: {data.get('error_details', 'N/A')}")
        print(f"  same: {data['true_summary'] == data['corrupted_summary']}")
        if i >= 5: break
```

## 2. Error Types mở rộng từ Research

Dựa trên paper , có các error types quan trọng khác:

| Error Type | Frequency | Mức độ nghiêm trọng | Có thể thêm? |
|------------|-----------|---------------------|--------------|
| **Fabrication** (tạo thông tin không có) | 43% | Cao | ❌ Khó synthetic |
| **Negation** (đã có) | 30% | **Cao nhất** | ✅ Đã có |
| **Contextual** (mix topics) | 17% | Trung bình | ⚠️ Khó implement |
| **Causality** (sai nguyên nhân) | 10% | Trung bình | ⚠️ Complex |
| **Omission** (bỏ sót) | 3.45% | Thấp | ❌ Khó synthetic |
| **Lab Value Error** | N/A | Cao | ✅ **Có thể thêm** |
| **Demographic Error** (age/sex) | Thấp | Trung bình | ✅ Dễ |

### Đề xuất thêm: Lab Value Corruption (LAB_VALUE)

```python
# Thêm vào ClinicalErrorSynthesizer

LAB_PATTERNS = [
    (re.compile(r'(\d+\.?\d*)\s*(mg/dL|g/dL|mmol/L|mEq/L|IU/L|U/L)\b', re.IGNORECASE), 'chemistry'),
    (re.compile(r'(\d+\.?\d*)\s*(K/uL|×10\^3/uL|×10\^9/L|cells/uL)\b', re.IGNORECASE), 'cbc'),
    (re.compile(r'(\d+\.?\d*)\s*%\b(?!\s*(?:mg|mcg|units))', re.IGNORECASE), 'percentage'),  # e.g., EF%, O2 sat
]

def _corrupt_lab_value(self, summary: str) -> Tuple[str, Optional[Dict]]:
    """Corrupt laboratory values by shifting within plausible ranges."""
    for pattern, lab_type in LAB_PATTERNS:
        matches = list(pattern.finditer(summary))
        if not matches:
            continue
            
        match = self.rng.choice(matches)
        original_val = float(match.group(1))
        unit = match.group(2)
        
        # Shift by ±20-50% (plausible but clinically significant)
        shift_pct = self.rng.choice([0.2, 0.3, 0.5])
        direction = self.rng.choice([-1, 1])
        new_val = original_val * (1 + direction * shift_pct)
        
        # Round appropriately
        if original_val >= 100:
            new_val = round(new_val)
        elif original_val >= 10:
            new_val = round(new_val, 1)
        else:
            new_val = round(new_val, 2)
            
        original_str = match.group(0)
        new_str = f"{new_val} {unit}"
        
        corrupted = summary[:match.start()] + new_str + summary[match.end():]
        return corrupted, {
            'type': 'LAB_VALUE',
            'lab_type': lab_type,
            'original': original_str,
            'corrupted': new_str,
        }
    
    return summary, None
```

### Đề xuất thêm: Demographic Corruption (DEMO)

```python
AGE_PATTERNS = [
    re.compile(r'\b(\d+)-year-old\b', re.IGNORECASE),
    re.compile(r'\bage\s*(\d+)\s*(?:yo|y\.o\.|years?)\b', re.IGNORECASE),
    re.compile(r'\b(\d+)\s*(?:yo|y\.o\.|years?)\s*(?:male|female|man|woman)\b', re.IGNORECASE),
]

SEX_PATTERNS = [
    (re.compile(r'\b(male|man|boy)\b', re.IGNORECASE), 'male'),
    (re.compile(r'\b(female|woman|girl)\b', re.IGNORECASE), 'female'),
]

def _corrupt_demographic(self, summary: str) -> Tuple[str, Optional[Dict]]:
    """Swap age or sex in patient description."""
    # Try age first
    for pattern in AGE_PATTERNS:
        match = pattern.search(summary)
        if match:
            original_age = int(match.group(1))
            # Shift by ±5-10 years
            shift = self.rng.choice([5, 10]) * self.rng.choice([-1, 1])
            new_age = max(18, min(100, original_age + shift))
            
            corrupted = summary[:match.start()] + summary[match.start():match.end()].replace(str(original_age), str(new_age)) + summary[match.end():]
            return corrupted, {
                'type': 'DEMO_AGE',
                'original': str(original_age),
                'corrupted': str(new_age),
            }
    
    # Try sex swap
    for pattern, sex in SEX_PATTERNS:
        match = pattern.search(summary)
        if match:
            opposite = 'female' if sex == 'male' else 'male'
            corrupted = pattern.sub(opposite, summary, count=1)
            return corrupted, {
                'type': 'DEMO_SEX',
                'original': sex,
                'corrupted': opposite,
            }
    
    return summary, None
```

## 3. Updated Error Distribution

```python
ERROR_TYPES = ['MED_NAME', 'MED_DOSE', 'TEMPORAL', 'NEGATION', 'PROCEDURE', 'LAB_VALUE', 'DEMO']

ERROR_WEIGHTS = {
    'MED_NAME': 0.25,    # Giảm từ 0.35
    'MED_DOSE': 0.20,    # Giảm từ 0.30  
    'TEMPORAL': 0.15,    # Giảm từ 0.20
    'NEGATION': 0.15,    # Giữ (quan trọng nhất theo research)
    'PROCEDURE': 0.10,    # Giữ
    'LAB_VALUE': 0.10,    # Mới
    'DEMO': 0.05,         # Mới (ít quan trọng hơn)
}
```

## 4. Khuyến nghị

| Hành động | Priority | Lý do |
|-----------|----------|-------|
| **Verify corruption đang hoạt động** | **Cao nhất** | Cần đảm bảo code sinh ra corrupted samples thực sự |
| **Thêm LAB_VALUE** | Cao | Phổ biến trong clinical notes, dễ implement |
| **Thêm DEMO** | Thấp | Ít ảnh hưởng đến clinical safety |
| **Bỏ/skip PROCEDURE** | Cân nhắc | Khó extract chính xác, ít samples bị ảnh hưởng |

**Kiểm tra ngay:**
```bash
python -c "
import json
count = {'corrupted': 0, 'clean': 0, 'types': {}}
with open('data/processed/train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('is_corrupted'):
            count['corrupted'] += 1
            for t in d.get('error_types', []):
                count['types'][t] = count['types'].get(t, 0) + 1
        else:
            count['clean'] += 1
print(f'Corrupted: {count[\"corrupted\"]}')
print(f'Clean: {count[\"clean\"]}')
print(f'Error types: {count[\"types\"]}')
"
```
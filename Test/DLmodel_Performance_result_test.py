import os
import torch
import pandas as pd
import numpy as np
import torch.quantization
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm

# 1. 환경 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_DATA_PATH = "D:\PJ\AI_Aegis_PJ(개인)\Data\model_data\Prompt_injection_dataset_v3.csv"  # 검증 데이터셋 경로 (prompt, label 컬럼 포함)

# 비교할 모델 버전 리스트 (각 버전이 저장된 폴더 경로)
MODEL_VERSIONS = {
    "v1_Base": "D:/PJ/AI_Aegis_PJ(개인)/model/Aegis_Ko_v2",
    "v3_FP_Refined": "D:/PJ/AI_Aegis_PJ(개인)/model/Aegis_Ko_v3",
    "v5_Domain_Added": "D:/PJ/AI_Aegis_PJ(개인)/model/Aegis_Ko_v5",
    "v7_Final_Pro": "D:/PJ/AI_Aegis_PJ(개인)/model/Aegis_Ko_v7"
}

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]

def evaluate_model(model_path, data):
    print(f"\n🔄 모델 로드 중: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cpu")
    model.eval()
    
    quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
    )
    
    labels = data['label'].tolist()
    texts = data['prompt'].tolist()
    predictions = []
    
    val_dataset = TextDataset(texts)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        # tqdm 대상을 'texts'가 아닌 'val_loader'로 바꿔야 배치 단위로 데이터가 나옵니다.
        for batch in tqdm(val_loader, desc="인퍼런스(Batch)"): 
            inputs = tokenizer(
                batch,  # 이제 반복문에서 나온 'batch'(32개의 문장 리스트)를 정상적으로 사용합니다.
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            ).to("cpu")
            
            # 양자화된 모델을 사용하여 예측
            outputs = quantized_model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).tolist()
            predictions.extend(preds)

    # 지표 계산
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

# 2. 메인 실행 로직
if __name__ == "__main__":
    # 데이터 로드
    df_val = pd.read_csv(VAL_DATA_PATH, encoding='utf-8-sig')   
    all_results = {}

    for version_name, path in MODEL_VERSIONS.items():
        if os.path.exists(path):
            result = evaluate_model(path, df_val)
            all_results[version_name] = result
        else:
            print(f"⚠️ 경로를 찾을 수 없음: {path}")

    # 3. 결과 비교 분석
    report_df = pd.DataFrame(all_results).T
    print("\n" + "="*50)
    print("📊 Aegis 모델 버전별 성능 비교 보고서")
    print("="*50)
    print(report_df)
    
    # CSV로 저장, 나중에 시각화에 활용
    report_df.to_csv("model_comparison_report.csv")
    print("\n✅ 결과가 'model_comparison_report.csv'로 저장되었습니다.")
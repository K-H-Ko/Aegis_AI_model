import pandas as pd

# 1. 데이터 로드
df = pd.read_csv('D:\PJ\AI_Aegis_PJ(개인)\Data\model_data\Prompt_injection_dataset_v3.csv', encoding='utf-8-sig') 

# 2. 라벨 개수 확인
counts = df['label'].value_counts()
percents = df['label'].value_counts(normalize=True) * 100

# 라벨 이름 매핑 정의 (라벨 2 추가)
label_names = {
    1: "🚨 공격(1)",
    0: "✅ 정상(0)",
    2: "⚠️ 애매(2)"
}

print("--- 🛡️ 데이터셋 라벨 분포 리포트 ---")
# 정해진 라벨 순서대로 출력 (0 -> 1 -> 2 순)
for label in sorted(label_names.keys()):
    if label in counts:
        name = label_names[label]
        print(f"{name}: {counts[label]:>5}개 ({percents[label]:.2f}%)")
    else:
        print(f"{label_names[label]}: 데이터 없음")
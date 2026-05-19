import pandas as pd

# load predictions
df = pd.read_csv("qa_predictions.csv")

# sort by lowest F1
df_sorted = df.sort_values(by="f1", ascending=True)

# show worst 20 examples
worst = df_sorted.head(20)

for _, row in worst.iterrows():

    print("=" * 80)

    print("QID:", row["qid"])
    print("Question:", row["question"])
    print("Gold Answer:", row["gold_answer"])
    print("Predicted Answer:", row["predicted_answer"])
    print("F1:", row["f1"])

    print()
import os
import numpy as np
import pandas as pd
import torch
import gradio as gr
from sklearn.preprocessing import StandardScaler

from model import LSTMOccupancyModel

MODEL_PATH = "lstm_occupancy.pth"

FEATURE_COLS_DEFAULT = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]


def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    feature_cols = bundle.get("feature_cols", FEATURE_COLS_DEFAULT)
    seq_length = int(bundle.get("seq_length", 30))

    # rebuild scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(bundle["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(bundle["scaler_scale"], dtype=np.float64)

    # rebuild model
    model = LSTMOccupancyModel(n_features=len(feature_cols), hidden_size=64)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    return model, scaler, feature_cols, seq_length


MODEL, SCALER, FEATURE_COLS, SEQ_LENGTH = load_bundle()


def load_df(split_name: str):
    """
    split_name: 'training' | 'test' | 'test2'
    """
    path_map = {
        "training": "data/datatraining.txt",
        "test": "data/datatest.txt",
        "test2": "data/datatest2.txt",
    }
    path = path_map.get(split_name)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file for split={split_name}: {path}")
    df = pd.read_csv(path)
    return df


@torch.no_grad()
def predict_probs_for_range(df: pd.DataFrame, start_idx: int, end_idx: int):
    """
    Returns probabilities and predicted labels for each timestep where a full window exists.
    We predict for indices [start_idx+SEQ_LENGTH .. end_idx] effectively.
    """
    X_raw = df[FEATURE_COLS].values.astype(np.float32)
    X = SCALER.transform(X_raw)

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    if start_idx < 0:
        start_idx = 0
    if end_idx > len(df) - 1:
        end_idx = len(df) - 1

    if end_idx - start_idx + 1 <= SEQ_LENGTH:
        raise ValueError(f"Range too small. Need at least SEQ_LENGTH+1 rows. SEQ_LENGTH={SEQ_LENGTH}")

    probs = []
    preds = []
    indices = []

    # slide within [start_idx, end_idx]
    for t in range(start_idx + SEQ_LENGTH, end_idx + 1):
        x_seq = X[t - SEQ_LENGTH:t]  # (T, F)
        x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0)  # (1,T,F)
        logits = MODEL(x_tensor)
        p = torch.sigmoid(logits).item()
        yhat = 1 if p >= 0.5 else 0

        probs.append(p)
        preds.append(yhat)
        indices.append(t)

    return np.array(indices), np.array(probs), np.array(preds)


def occupancy_report(split_name, start_idx, end_idx):
    df = load_df(split_name)

    # Compute occupancy ratio on predicted labels
    idxs, probs, preds = predict_probs_for_range(df, start_idx, end_idx)

    ratio = preds.mean() * 100.0  # percent
    avg_prob = probs.mean() * 100.0

    # A small preview table
    preview = df.loc[idxs[:10], ["date"] + FEATURE_COLS].copy()
    preview["pred_prob(occupied)"] = probs[:10]
    preview["pred_label"] = preds[:10]

    text = (
        f"Split: {split_name}\n"
        f"Index range used for predictions: {idxs[0]}..{idxs[-1]} (total {len(idxs)} predictions)\n"
        f"SEQ_LENGTH: {SEQ_LENGTH}\n\n"
        f"Estimated Occupancy Ratio (by labels >=0.5): {ratio:.2f}%\n"
        f"Average Occupied Probability: {avg_prob:.2f}%\n"
    )

    return text, preview


def build_ui():
    with gr.Blocks(title="Room Occupancy Ratio - LSTM") as demo:
        gr.Markdown("# Room/House Occupancy Ratio Estimation (LSTM)")
        gr.Markdown(
            f"- Features: {', '.join(FEATURE_COLS)}\n"
            f"- Window length (SEQ_LENGTH): **{SEQ_LENGTH}**\n"
            "Bu arayüz, seçtiğin dataset parçasında belirli bir indeks aralığı için **doluluk oranını (%)** hesaplar."
        )

        with gr.Row():
            split = gr.Dropdown(choices=["training", "test", "test2"], value="test2", label="Dataset split")
            start_idx = gr.Number(value=0, precision=0, label="Start index")
            end_idx = gr.Number(value=500, precision=0, label="End index")

        btn = gr.Button("Compute occupancy ratio")

        out_text = gr.Textbox(label="Report", lines=8)
        out_table = gr.Dataframe(label="Preview (first 10 predictions)")

        btn.click(fn=occupancy_report, inputs=[split, start_idx, end_idx], outputs=[out_text, out_table])

        gr.Markdown(
            "İpucu: Aralığı büyütmek (örn. 0..1000) daha stabil bir oran verir. "
            "Aralık en az SEQ_LENGTH+1 satır olmalı."
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True)

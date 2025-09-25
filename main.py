from scapy.all import sniff, TCP
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def compute_entropy(window):
    try:
        ports = [p[TCP].sport for p in window if p.haslayer(TCP)]
        if not ports:
            return 0
        probs = np.bincount(ports) / len(ports)
        return -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
    except Exception as e:
        logging.error(f"Entropy error: {e}")
        return 0


def adaptive_window(packets, size=50):
    if len(packets) < 10:
        return np.array([]), []
    features, anomalies = [], []
    for i in range(0, len(packets), size // 2):
        window = packets[i : i + size]
        if len(window) < 10:
            break
        try:
            inter_arrivals = [
                window[j + 1].time - window[j].time for j in range(len(window) - 1)
            ]
            entropy = compute_entropy(window)
            mean_arrival = np.mean(inter_arrivals) if inter_arrivals else 0
            variance = np.var(inter_arrivals) if inter_arrivals else 0
            features.append([mean_arrival, entropy])
            size = max(10, size // 2) if variance > 0.2 else min(100, int(size * 1.5))
            cusum = (
                sum(abs(ia - mean_arrival) for ia in inter_arrivals)
                if inter_arrivals
                else 0
            )
            if cusum > 0.3:
                anomalies.append(i)
        except Exception as e:
            logging.error(f"Window error at {i}: {e}")
            continue
    if not features:
        return np.array([]), []
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    labels = [1 if i in anomalies else 0 for i in range(len(features))]
    model = DecisionTreeClassifier(max_depth=3).fit(scaled_features, labels)
    predictions = model.predict(scaled_features)
    return scaled_features, predictions


def main(interface="eth0"):
    try:
        logging.info(f"Capturing 100 TCP packets on {interface}")
        packets = sniff(iface=interface, count=100, filter="tcp", timeout=10)
        if not packets:
            return
        features, predictions = adaptive_window(packets)
        if features.size == 0:
            return
        for i, (feat, pred) in enumerate(zip(features, predictions)):
            status = "LDoS detected" if pred == 1 else "Normal"
            logging.info(f"Window {i}: Features={feat}, Status={status}")
        logging.info("LDoS attack detected!" if any(predictions) else "No LDoS attacks")
    except Exception as e:
        logging.error(f"Main error: {e}")


if __name__ == "__main__":
    main(interface="wlp0s20f3")

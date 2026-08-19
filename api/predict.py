from http.server import BaseHTTPRequestHandler
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            from src.pipeline.predict_pipeline import CustomData, PredictPipeline

            data = CustomData(
                LIMIT_BAL=float(body["LIMIT_BAL"]),
                SEX=int(body["SEX"]),
                EDUCATION=int(body["EDUCATION"]),
                MARRIAGE=int(body["MARRIAGE"]),
                AGE=int(body["AGE"]),
                PAY_0=int(body["PAY_0"]),
                PAY_2=int(body["PAY_2"]),
                PAY_3=int(body["PAY_3"]),
                PAY_4=int(body["PAY_4"]),
                PAY_5=int(body["PAY_5"]),
                PAY_6=int(body["PAY_6"]),
                BILL_AMT1=float(body["BILL_AMT1"]),
                BILL_AMT2=float(body["BILL_AMT2"]),
                BILL_AMT3=float(body["BILL_AMT3"]),
                BILL_AMT4=float(body["BILL_AMT4"]),
                BILL_AMT5=float(body["BILL_AMT5"]),
                BILL_AMT6=float(body["BILL_AMT6"]),
                PAY_AMT1=float(body["PAY_AMT1"]),
                PAY_AMT2=float(body["PAY_AMT2"]),
                PAY_AMT3=float(body["PAY_AMT3"]),
                PAY_AMT4=float(body["PAY_AMT4"]),
                PAY_AMT5=float(body["PAY_AMT5"]),
                PAY_AMT6=float(body["PAY_AMT6"]),
            )

            pred_df = data.get_data_as_data_frame()
            pipeline = PredictPipeline()
            preds, proba = pipeline.predict(pred_df)

            prob = float(proba[0]) * 100
            prediction = "Will Default" if preds[0] == 1 else "Will Not Default"

            if prob >= 70:
                risk_level, risk_color = "High Risk", "high"
            elif prob >= 40:
                risk_level, risk_color = "Moderate Risk", "moderate"
            else:
                risk_level, risk_color = "Low Risk", "low"

            result = {
                "prediction": prediction,
                "probability": round(prob, 1),
                "risk_level": risk_level,
                "risk_color": risk_color,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            import traceback
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e), "trace": traceback.format_exc()}).encode())

    def log_message(self, format, *args):
        pass

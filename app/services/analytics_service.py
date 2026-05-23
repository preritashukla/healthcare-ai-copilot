import pandas as pd
import numpy as np

class AnalyticsService:
    @staticmethod
    def get_patient_vitals(patient_id: str = "Marcus Vance") -> dict:
        """Processes vital sign telemetry trends and computes clinical summaries."""
        # Baseline dataset mirroring original RAG python script
        data = {
            "Day": [1, 2, 3, 4, 5],
            "HeartRate": [85, 90, 95, 100, 92],
            "Temperature": [98.4, 98.7, 99.1, 99.5, 98.9]
        }
        
        df = pd.DataFrame(data)
        
        # Calculate statistics
        hr_min = int(df["HeartRate"].min())
        hr_max = int(df["HeartRate"].max())
        hr_avg = float(df["HeartRate"].mean())
        
        temp_min = float(df["Temperature"].min())
        temp_max = float(df["Temperature"].max())
        temp_avg = float(df["Temperature"].mean())
        
        # Set clinical alerts based on thresholds (HR > 95 or Temp > 99.0)
        vitals_list = []
        is_unstable = False
        for idx, row in df.iterrows():
            day_str = f"Day {int(row['Day'])}"
            hr = int(row['HeartRate'])
            temp = float(row['Temperature'])
            
            alerts = []
            if hr > 95:
                alerts.append("Tachycardia")
                is_unstable = True
            if temp > 99.0:
                alerts.append("Low-grade fever")
                is_unstable = True
                
            vitals_list.append({
                "day": day_str,
                "heartRate": hr,
                "temp": temp,
                "alerts": alerts,
                "status": "warning" if alerts else "stable"
            })
            
        summary = {
            "patient_id": patient_id,
            "status": "Monitoring Required" if is_unstable else "Stable",
            "telemetry": vitals_list,
            "statistics": {
                "heartRate": {
                    "min": hr_min,
                    "max": hr_max,
                    "avg": round(hr_avg, 2),
                    "status": "Elevated" if hr_avg > 90 else "Normal"
                },
                "temperature": {
                    "min": temp_min,
                    "max": temp_max,
                    "avg": round(temp_avg, 2),
                    "status": "Low-grade fever" if temp_max > 99.0 else "Normal"
                }
            }
        }
        
        return summary

analytics_service = AnalyticsService()

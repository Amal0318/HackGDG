// Utility functions for medical calculations

export function calculateMAP(systolic: number, diastolic: number): number {
  return Math.round((systolic + 2 * diastolic) / 3);
}

export function calculateRiskScore(vitals: {
  heartRate: number;
  systolicBP: number;
  diastolicBP: number;
  spo2: number;
  respiratoryRate: number;
  temperature: number;
  gcs: number;
}): number {
  let score = 0;
  
  // Heart rate deviations
  if (vitals.heartRate < 60) score += (60 - vitals.heartRate) / 2;
  if (vitals.heartRate > 100) score += (vitals.heartRate - 100) / 3;
  if (vitals.heartRate > 140) score += 20;
  
  // Blood pressure
  if (vitals.systolicBP < 90) score += (90 - vitals.systolicBP) / 2;
  if (vitals.systolicBP > 180) score += (vitals.systolicBP - 180) / 3;
  
  // SpO2
  if (vitals.spo2 < 95) score += (95 - vitals.spo2) * 2;
  if (vitals.spo2 < 90) score += 15;
  
  // Respiratory rate
  if (vitals.respiratoryRate < 12 || vitals.respiratoryRate > 20) {
    score += Math.abs(16 - vitals.respiratoryRate);
  }
  if (vitals.respiratoryRate > 30) score += 15;
  
  // Temperature
  if (vitals.temperature < 36 || vitals.temperature > 38) {
    score += Math.abs(37 - vitals.temperature) * 5;
  }
  
  // GCS
  if (vitals.gcs < 15) score += (15 - vitals.gcs) * 3;
  if (vitals.gcs <= 8) score += 20;
  
  return Math.min(Math.round(score), 100);
}

export function getRiskLevel(score: number): 'low' | 'medium' | 'high' | 'critical' {
  if (score <= 30) return 'low';
  if (score <= 50) return 'medium';
  if (score <= 70) return 'high';
  return 'critical';
}

export function getVitalStatus(value: number, normalRange: [number, number]): 'normal' | 'warning' | 'critical' {
  const [min, max] = normalRange;
  const warningThreshold = (max - min) * 0.15;
  
  if (value < min - warningThreshold || value > max + warningThreshold) {
    return 'critical';
  }
  if (value < min || value > max) {
    return 'warning';
  }
  return 'normal';
}

export function generateTrendData(baseValue: number, variance: number, points: number = 48): number[] {
  const data: number[] = [];
  let current = baseValue;
  
  for (let i = 0; i < points; i++) {
    const change = (Math.random() - 0.5) * variance;
    current = Math.max(0, current + change);
    data.push(Math.round(current * 10) / 10);
  }
  
  return data;
}

export function predictNextValues(history: number[], futurePoints: number = 8): number[] {
  if (history.length < 3) return [];
  
  // Simple linear regression for prediction
  const recentHistory = history.slice(-10);
  const trend = (recentHistory[recentHistory.length - 1] - recentHistory[0]) / recentHistory.length;
  const predicted: number[] = [];
  
  let lastValue = history[history.length - 1];
  for (let i = 0; i < futurePoints; i++) {
    lastValue += trend;
    predicted.push(Math.round(lastValue * 10) / 10);
  }
  
  return predicted;
}

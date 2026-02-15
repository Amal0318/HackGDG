import { calculateMAP, calculateRiskScore, generateTrendData, predictNextValues } from './utils/calculations';

export interface VitalSigns {
  heartRate: number;
  systolicBP: number;
  diastolicBP: number;
  spo2: number;
  respiratoryRate: number;
  temperature: number;
  gcs: number;
  cvp: number;
  map: number;
  urine: number;
  lactate: number;
  glucose: number;
  fio2: number;
  peep: number;
}

export interface Alert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  vital?: string;
  timestamp: Date;
  acknowledged: boolean;
}

export interface Intervention {
  id: string;
  type: string;
  timestamp: Date;
  user: string;
  details: string;
}

export interface TrendData {
  heartRate: number[];
  systolicBP: number[];
  spo2: number[];
  respiratoryRate: number[];
  temperature: number[];
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  bed: string;
  floor: number;
  ward: string;
  assignedDoctor: string;
  assignedNurse: string;
  admissionDate: Date;
  vitals: VitalSigns;
  riskScore: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  trends: TrendData;
  predictive: {
    heartRate: number[];
    spo2: number[];
    riskTrend: 'improving' | 'stable' | 'deteriorating';
    estimatedTimeToAlert: number | null; // minutes
  };
  alerts: Alert[];
  interventions: Intervention[];
  diagnosis: string;
}

const patientNames = [
  'Sarah Johnson', 'Michael Chen', 'Emma Williams', 'James Davis',
  'Olivia Brown', 'Robert Garcia', 'Sophia Martinez', 'William Rodriguez',
  'Ava Lopez', 'David Anderson', 'Isabella Thomas', 'John Jackson',
  'Mia White', 'Daniel Harris', 'Charlotte Martin', 'Joseph Thompson'
];

const doctors = ['Dr. Anderson', 'Dr. Patel', 'Dr. Kim', 'Dr. Johnson', 'Dr. Martinez'];
const nurses = ['Nurse Williams', 'Nurse Chen', 'Nurse Rodriguez', 'Nurse Thompson'];

const diagnoses = [
  'Post-operative recovery',
  'Septic shock',
  'Acute respiratory distress',
  'Cardiac arrhythmia',
  'Multi-organ dysfunction',
  'Pneumonia',
  'Post-cardiac arrest',
  'Traumatic brain injury'
];

function createVitals(severity: 'low' | 'medium' | 'high' | 'critical'): VitalSigns {
  let heartRate, systolicBP, diastolicBP, spo2, respiratoryRate, temperature, gcs;
  
  switch (severity) {
    case 'critical':
      heartRate = 135 + Math.random() * 30;
      systolicBP = 85 + Math.random() * 15;
      diastolicBP = 45 + Math.random() * 15;
      spo2 = 86 + Math.random() * 6;
      respiratoryRate = 28 + Math.random() * 10;
      temperature = 38.5 + Math.random() * 1.5;
      gcs = 8 + Math.floor(Math.random() * 4);
      break;
    case 'high':
      heartRate = 110 + Math.random() * 20;
      systolicBP = 95 + Math.random() * 20;
      diastolicBP = 55 + Math.random() * 15;
      spo2 = 91 + Math.random() * 4;
      respiratoryRate = 22 + Math.random() * 6;
      temperature = 37.8 + Math.random() * 1;
      gcs = 11 + Math.floor(Math.random() * 3);
      break;
    case 'medium':
      heartRate = 85 + Math.random() * 20;
      systolicBP = 110 + Math.random() * 20;
      diastolicBP = 65 + Math.random() * 15;
      spo2 = 94 + Math.random() * 3;
      respiratoryRate = 18 + Math.random() * 4;
      temperature = 37.2 + Math.random() * 0.8;
      gcs = 13 + Math.floor(Math.random() * 2);
      break;
    default: // low
      heartRate = 70 + Math.random() * 15;
      systolicBP = 115 + Math.random() * 15;
      diastolicBP = 70 + Math.random() * 10;
      spo2 = 96 + Math.random() * 3;
      respiratoryRate = 14 + Math.random() * 4;
      temperature = 36.8 + Math.random() * 0.6;
      gcs = 15;
  }
  
  return {
    heartRate: Math.round(heartRate),
    systolicBP: Math.round(systolicBP),
    diastolicBP: Math.round(diastolicBP),
    spo2: Math.round(spo2 * 10) / 10,
    respiratoryRate: Math.round(respiratoryRate),
    temperature: Math.round(temperature * 10) / 10,
    gcs,
    cvp: Math.round((4 + Math.random() * 8) * 10) / 10,
    map: 0, // Will be calculated
    urine: Math.round(40 + Math.random() * 80),
    lactate: Math.round((0.5 + Math.random() * (severity === 'critical' ? 6 : 2)) * 10) / 10,
    glucose: Math.round(80 + Math.random() * 150),
    fio2: Math.round(21 + Math.random() * (severity === 'critical' ? 60 : 30)),
    peep: Math.round(Math.random() * (severity === 'critical' ? 12 : 6))
  };
}

function createAlerts(vitals: VitalSigns): Alert[] {
  const alerts: Alert[] = [];
  const now = new Date();
  
  if (vitals.heartRate > 120) {
    alerts.push({
      id: `alert-hr-${Math.random()}`,
      severity: vitals.heartRate > 140 ? 'critical' : 'warning',
      message: `Tachycardia detected: ${vitals.heartRate} bpm`,
      vital: 'Heart Rate',
      timestamp: new Date(now.getTime() - Math.random() * 3600000),
      acknowledged: false
    });
  }
  
  if (vitals.spo2 < 92) {
    alerts.push({
      id: `alert-spo2-${Math.random()}`,
      severity: vitals.spo2 < 88 ? 'critical' : 'warning',
      message: `Low oxygen saturation: ${vitals.spo2}%`,
      vital: 'SpO2',
      timestamp: new Date(now.getTime() - Math.random() * 3600000),
      acknowledged: false
    });
  }
  
  if (vitals.systolicBP < 100) {
    alerts.push({
      id: `alert-bp-${Math.random()}`,
      severity: vitals.systolicBP < 90 ? 'critical' : 'warning',
      message: `Hypotension: ${vitals.systolicBP}/${vitals.diastolicBP} mmHg`,
      vital: 'Blood Pressure',
      timestamp: new Date(now.getTime() - Math.random() * 3600000),
      acknowledged: false
    });
  }
  
  if (vitals.respiratoryRate > 24) {
    alerts.push({
      id: `alert-rr-${Math.random()}`,
      severity: vitals.respiratoryRate > 30 ? 'critical' : 'warning',
      message: `Tachypnea: ${vitals.respiratoryRate} breaths/min`,
      vital: 'Respiratory Rate',
      timestamp: new Date(now.getTime() - Math.random() * 3600000),
      acknowledged: false
    });
  }
  
  if (vitals.gcs < 13) {
    alerts.push({
      id: `alert-gcs-${Math.random()}`,
      severity: vitals.gcs <= 8 ? 'critical' : 'warning',
      message: `Decreased consciousness: GCS ${vitals.gcs}`,
      vital: 'GCS',
      timestamp: new Date(now.getTime() - Math.random() * 3600000),
      acknowledged: false
    });
  }
  
  return alerts;
}

function createInterventions(): Intervention[] {
  const types = ['Fluid bolus', 'Medication', 'O2 adjustment', 'Position change', 'Lab draw'];
  const now = new Date();
  const interventions: Intervention[] = [];
  
  for (let i = 0; i < 3 + Math.floor(Math.random() * 4); i++) {
    const type = types[Math.floor(Math.random() * types.length)];
    interventions.push({
      id: `int-${Math.random()}`,
      type,
      timestamp: new Date(now.getTime() - Math.random() * 86400000),
      user: nurses[Math.floor(Math.random() * nurses.length)],
      details: type === 'Fluid bolus' ? '500ml NS' : 
               type === 'Medication' ? 'Norepinephrine 0.05 mcg/kg/min' :
               type === 'O2 adjustment' ? 'FiO2 increased to 60%' :
               type === 'Position change' ? 'Prone positioning' :
               'Complete metabolic panel'
    });
  }
  
  return interventions.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
}

// Generate 16 patients across 3 floors
export const patients: Patient[] = [];

const riskDistribution = ['low', 'low', 'low', 'low', 'medium', 'medium', 'medium', 'medium', 'medium', 'high', 'high', 'high', 'critical', 'critical', 'critical', 'critical'] as ('low' | 'medium' | 'high' | 'critical')[];

for (let i = 0; i < 16; i++) {
  const floor = 1 + Math.floor(i / 6); // Distribute across 3 floors
  const wardIndex = Math.floor((i % 6) / 2);
  const ward = String.fromCharCode(65 + wardIndex); // A, B, C, D
  const bedNum = (i % 2) + 1;
  const bed = `${floor}${ward}-${bedNum}`;
  
  const severity = riskDistribution[i];
  const vitals = createVitals(severity);
  vitals.map = calculateMAP(vitals.systolicBP, vitals.diastolicBP);
  
  const riskScore = calculateRiskScore(vitals);
  const alerts = createAlerts(vitals);
  
  const trends = {
    heartRate: generateTrendData(vitals.heartRate, 5),
    systolicBP: generateTrendData(vitals.systolicBP, 8),
    spo2: generateTrendData(vitals.spo2, 2),
    respiratoryRate: generateTrendData(vitals.respiratoryRate, 2),
    temperature: generateTrendData(vitals.temperature, 0.3)
  };
  
  const predictedHR = predictNextValues(trends.heartRate);
  const predictedSpO2 = predictNextValues(trends.spo2);
  
  // Determine if patient is deteriorating
  const hrTrend = trends.heartRate.slice(-5);
  const hrIncreasing = hrTrend[hrTrend.length - 1] > hrTrend[0] + 5;
  const spo2Decreasing = trends.spo2.slice(-5)[4] < trends.spo2.slice(-5)[0] - 2;
  
  let riskTrend: 'improving' | 'stable' | 'deteriorating' = 'stable';
  let estimatedTimeToAlert: number | null = null;
  
  if (hrIncreasing && spo2Decreasing && severity === 'high') {
    riskTrend = 'deteriorating';
    estimatedTimeToAlert = 15 + Math.floor(Math.random() * 10);
  } else if (hrIncreasing || spo2Decreasing) {
    riskTrend = 'deteriorating';
    estimatedTimeToAlert = null;
  } else if (severity === 'medium' || severity === 'low') {
    riskTrend = 'improving';
  }
  
  patients.push({
    id: `P${String(i + 1).padStart(3, '0')}`,
    name: patientNames[i],
    age: 35 + Math.floor(Math.random() * 45),
    bed,
    floor,
    ward,
    assignedDoctor: doctors[Math.floor(Math.random() * doctors.length)],
    assignedNurse: nurses[Math.floor(Math.random() * nurses.length)],
    admissionDate: new Date(Date.now() - Math.random() * 7 * 86400000),
    vitals,
    riskScore,
    riskLevel: severity,
    trends,
    predictive: {
      heartRate: predictedHR,
      spo2: predictedSpO2,
      riskTrend,
      estimatedTimeToAlert
    },
    alerts,
    interventions: createInterventions(),
    diagnosis: diagnoses[Math.floor(Math.random() * diagnoses.length)]
  });
}

// Helper functions
export function getPatientsByFloor(floor: number): Patient[] {
  return patients.filter(p => p.floor === floor);
}

export function getPatientsByWard(floor: number, ward: string): Patient[] {
  return patients.filter(p => p.floor === floor && p.ward === ward);
}

export function getPatientsByDoctor(doctorName: string): Patient[] {
  return patients.filter(p => p.assignedDoctor === doctorName);
}

export function getFloorStats(floor: number) {
  const floorPatients = getPatientsByFloor(floor);
  return {
    total: floorPatients.length,
    highRisk: floorPatients.filter(p => p.riskLevel === 'high' || p.riskLevel === 'critical').length,
    activeAlerts: floorPatients.reduce((sum, p) => sum + p.alerts.length, 0),
    avgRiskScore: Math.round(floorPatients.reduce((sum, p) => sum + p.riskScore, 0) / floorPatients.length)
  };
}

export function getAllAlerts(): Alert[] {
  return patients.flatMap(p => p.alerts.map(a => ({ ...a, patientName: p.name, patientBed: p.bed })));
}

export function getSystemStats() {
  const totalBeds = 48;
  const occupiedBeds = patients.length;
  const criticalAlerts = patients.reduce((sum, p) => sum + p.alerts.filter(a => a.severity === 'critical').length, 0);
  const avgRiskScore = Math.round(patients.reduce((sum, p) => sum + p.riskScore, 0) / patients.length);
  
  return {
    totalPatients: patients.length,
    totalBeds,
    occupiedBeds,
    bedOccupancy: Math.round((occupiedBeds / totalBeds) * 100),
    criticalAlerts,
    avgRiskScore,
    riskDistribution: {
      low: patients.filter(p => p.riskLevel === 'low').length,
      medium: patients.filter(p => p.riskLevel === 'medium').length,
      high: patients.filter(p => p.riskLevel === 'high').length,
      critical: patients.filter(p => p.riskLevel === 'critical').length
    }
  };
}

// Generate historical data for charts (24 hours)
// Based on realistic ICU patterns: more alerts during shift changes (7am, 3pm, 11pm)
// and rounds (8-10am), fewer during night hours
export function generateAlertHistory(currentPatientCount: number = 24, avgRiskScore: number = 50) {
  const data = [];
  const now = new Date();
  
  for (let i = 23; i >= 0; i--) {
    const hour = new Date(now.getTime() - i * 3600000);
    const hourOfDay = hour.getHours();
    
    // Base alert rate depends on patient count and risk
    const riskMultiplier = avgRiskScore / 50; // Higher avg risk = more alerts
    const patientMultiplier = currentPatientCount / 24; // More patients = more alerts
    
    // Time-based patterns (ICU activity patterns)
    let timeMultiplier = 1.0;
    if (hourOfDay >= 7 && hourOfDay <= 9) {
      // Morning rounds and shift change - high activity
      timeMultiplier = 1.8;
    } else if (hourOfDay >= 14 && hourOfDay <= 16) {
      // Afternoon shift change - moderate activity
      timeMultiplier = 1.5;
    } else if (hourOfDay >= 22 || hourOfDay <= 1) {
      // Night shift change - moderate activity
      timeMultiplier = 1.3;
    } else if (hourOfDay >= 2 && hourOfDay <= 5) {
      // Deep night - lowest activity
      timeMultiplier = 0.5;
    }
    
    // Calculate alerts with realistic patterns
    const baseCritical = 2 * riskMultiplier * patientMultiplier * timeMultiplier;
    const baseWarning = 8 * riskMultiplier * patientMultiplier * timeMultiplier;
    const baseInfo = 5 * patientMultiplier * timeMultiplier;
    
    // Add some variability but keep it realistic
    const critical = Math.max(0, Math.floor(baseCritical + (Math.random() - 0.5) * 2));
    const warning = Math.max(0, Math.floor(baseWarning + (Math.random() - 0.5) * 4));
    const info = Math.max(0, Math.floor(baseInfo + (Math.random() - 0.5) * 3));
    
    data.push({
      hour: hourOfDay,
      critical,
      warning,
      info
    });
  }
  
  return data;
}

// Generate 7-day trend data for sparklines
export function generate7DayTrend(currentValue: number) {
  const data = [];
  let value = currentValue - (Math.random() * 10 - 5);
  
  for (let i = 0; i < 7; i++) {
    data.push(Math.round(value));
    value += Math.random() * 6 - 3;
  }
  
  return data;
}

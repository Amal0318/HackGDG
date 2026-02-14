import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useStore } from '../../store';

interface ComparativeChartProps {
  patientIds: string[];
  metric: 'heart_rate' | 'spo2' | 'systolic_bp' | 'risk_score';
}

export const ComparativeChart = ({ patientIds, metric }: ComparativeChartProps) => {
  const patients = useStore(state => state.patients);
  
  // Build comparative data
  const data = buildComparativeData(patients, patientIds, metric);
  
  if (data.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-6 text-center text-gray-500">
        No data available for comparison
      </div>
    );
  }
  
  const metricConfig = getMetricConfig(metric);
  const colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
  
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <h3 className="text-gray-900 font-semibold mb-4">
        📊 Comparative {metricConfig.label}
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="timestamp" 
            stroke="#6b7280"
            tick={{ fill: '#6b7280', fontSize: 12 }}
            tickFormatter={(value) => new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          />
          <YAxis 
            stroke="#6b7280"
            tick={{ fill: '#6b7280', fontSize: 12 }}
            domain={metricConfig.domain}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            labelFormatter={(value) => new Date(value).toLocaleString()}
          />
          <Legend />
          {patientIds.map((patientId, index) => (
            <Line
              key={patientId}
              type="monotone"
              dataKey={patientId}
              stroke={colors[index % colors.length]}
              strokeWidth={2}
              dot={false}
              name={patientId}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

function buildComparativeData(
  patients: Map<string, any>,
  patientIds: string[],
  metric: string
): any[] {
  // Get all unique timestamps
  const timestampSet = new Set<string>();
  
  patientIds.forEach(id => {
    const patientData = patients.get(id);
    if (patientData?.history) {
      patientData.history.forEach((vital: any) => {
        timestampSet.add(vital.timestamp);
      });
    }
  });
  
  const timestamps = Array.from(timestampSet).sort();
  
  // Build data points
  return timestamps.slice(-20).map(timestamp => {
    const point: any = { timestamp };
    
    patientIds.forEach(id => {
      const patientData = patients.get(id);
      if (patientData?.history) {
        const vital = patientData.history.find((v: any) => v.timestamp === timestamp);
        if (vital) {
          point[id] = vital[metric];
        }
      }
    });
    
    return point;
  });
}

function getMetricConfig(metric: string) {
  switch (metric) {
    case 'heart_rate':
      return { label: 'Heart Rate', domain: [40, 140] };
    case 'spo2':
      return { label: 'SpO2', domain: [90, 100] };
    case 'systolic_bp':
      return { label: 'Systolic BP', domain: [60, 180] };
    case 'risk_score':
      return { label: 'Risk Score', domain: [0, 100] };
    default:
      return { label: 'Unknown', domain: [0, 100] };
  }
}

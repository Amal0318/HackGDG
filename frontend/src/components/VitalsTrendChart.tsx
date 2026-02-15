import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { format } from 'date-fns';

interface VitalsDataPoint {
  timestamp: string;
  heart_rate: number;
  systolic_bp: number;
  diastolic_bp: number;
  spo2: number;
  respiratory_rate: number;
  temperature: number;
}

interface VitalsTrendChartProps {
  data: VitalsDataPoint[];
  height?: number;
  showGrid?: boolean;
  selectedVitals?: string[];
}

export default function VitalsTrendChart({ 
  data, 
  height = 300, 
  showGrid = true,
  selectedVitals = ['heart_rate', 'systolic_bp', 'spo2']
}: VitalsTrendChartProps) {
  
  const chartData = useMemo(() => {
    return data.map(point => ({
      time: format(new Date(point.timestamp), 'HH:mm:ss'),
      fullTime: point.timestamp,
      heartRate: point.heart_rate,
      systolicBp: point.systolic_bp,
      diastolicBp: point.diastolic_bp,
      spo2: point.spo2,
      respRate: point.respiratory_rate,
      temp: point.temperature,
    }));
  }, [data]);

  const vitalKeyMap: Record<string, keyof typeof chartData[0]> = {
    heart_rate: 'heartRate',
    systolic_bp: 'systolicBp',
    diastolic_bp: 'diastolicBp',
    spo2: 'spo2',
    respiratory_rate: 'respRate',
    temperature: 'temp',
  };

  const vitalConfigs = {
    heartRate: {
      color: '#ef4444',
      label: 'Heart Rate (bpm)',
      normalRange: [60, 100]
    },
    systolicBp: {
      color: '#3b82f6',
      label: 'Systolic BP (mmHg)', 
      normalRange: [90, 140]
    },
    diastolicBp: {
      color: '#06b6d4',
      label: 'Diastolic BP (mmHg)',
      normalRange: [60, 90]
    },
    spo2: {
      color: '#10b981',
      label: 'SpO₂ (%)',
      normalRange: [95, 100]
    },
    respRate: {
      color: '#f59e0b',
      label: 'Resp Rate (br/min)',
      normalRange: [12, 20]
    },
    temp: {
      color: '#8b5cf6',
      label: 'Temperature (°C)',
      normalRange: [36.1, 37.2]
    }
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="text-xs text-gray-600 mb-2">{label}</p>
          {payload.map((entry: any, index: number) => {
            const config = vitalConfigs[entry.dataKey as keyof typeof vitalConfigs];
            return (
              <p key={index} className="text-sm font-medium" style={{ color: entry.color }}>
                {config.label}: {entry.value?.toFixed(1)}
              </p>
            );
          })}
        </div>
      );
    }
    return null;
  };

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        No vitals data available
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Stats Summary */}
      <div className="grid grid-cols-3 gap-4 text-xs">
        {selectedVitals.slice(0, 3).map((vital) => {
          const mappedKey = vitalKeyMap[vital] || (vital as keyof typeof chartData[0]);
          const config = vitalConfigs[mappedKey as keyof typeof vitalConfigs] || {
            color: '#6b7280',
            label: vital,
            normalRange: [0, 0],
          };
          const latestValue = chartData[chartData.length - 1]?.[mappedKey];
          const numericLatest = typeof latestValue === 'number'
            ? latestValue
            : Number(latestValue ?? 0);
          const average = chartData.reduce((sum, d) => sum + (d[mappedKey] as number || 0), 0) / chartData.length;
          
          return (
            <div key={vital} className="p-2 bg-gray-50 rounded">
              <span className="text-gray-500 block">{config.label}</span>
              <span className="font-semibold" style={{ color: config.color }}>
                Current: {numericLatest.toFixed(1)}
              </span>
              <span className="text-gray-500 block text-xs">
                Avg: {average.toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />}
          
          <XAxis 
            dataKey="time" 
            tick={{ fontSize: 10, fill: '#9ca3af' }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <YAxis 
            tick={{ fontSize: 10, fill: '#9ca3af' }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            domain={['auto', 'auto']}
            label={{ value: 'Values', angle: -90, position: 'insideLeft', style: { fontSize: 10, fill: '#6b7280' } }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ fontSize: '12px' }}
            iconType="line"
          />
          
          {selectedVitals.map((vital) => {
            const mappedKey = vitalKeyMap[vital] || (vital as keyof typeof chartData[0]);
            const config = vitalConfigs[mappedKey as keyof typeof vitalConfigs] || {
              color: '#6b7280',
              label: vital,
              normalRange: [0, 0],
            };
            return (
              <Line 
                key={vital}
                type="monotone" 
                dataKey={mappedKey as string}
                stroke={config.color} 
                strokeWidth={2}
                dot={{ r: 2 }}
                activeDot={{ r: 4 }}
                name={config.label}
                isAnimationActive={false}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
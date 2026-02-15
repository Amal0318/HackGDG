import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { format } from 'date-fns';

interface RiskDataPoint {
  timestamp: string;
  risk_score: number;
}

interface RiskTrendChartProps {
  data: RiskDataPoint[];
  height?: number;
  showGrid?: boolean;
}

export default function RiskTrendChart({ 
  data, 
  height = 200, 
  showGrid = true 
}: RiskTrendChartProps) {
  
  const chartData = useMemo(() => {
    return data.map(point => {
      // Risk score might be 0-1 (backend raw) or 0-100 (already transformed)
      // Detect and normalize
      const riskValue = point.risk_score > 10 ? point.risk_score : point.risk_score * 100;
      return {
        time: format(new Date(point.timestamp), 'HH:mm:ss'),
        fullTime: point.timestamp,
        risk: riskValue.toFixed(1), // Show as percentage with 1 decimal
        riskValue: riskValue,
      };
    });
  }, [data]);

  const getRiskColor = (risk: number) => {
    if (risk <= 30) return '#10b981'; // green
    if (risk <= 60) return '#f59e0b'; // yellow
    if (risk <= 80) return '#f97316'; // orange
    return '#ef4444'; // red
  };

  const averageRisk = useMemo(() => {
    if (chartData.length === 0) return 0;
    return chartData.reduce((sum, d) => sum + d.riskValue, 0) / chartData.length;
  }, [chartData]);

  const maxRisk = useMemo(() => {
    if (chartData.length === 0) return 0;
    return Math.max(...chartData.map(d => d.riskValue));
  }, [chartData]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="text-xs text-gray-600">{data.time}</p>
          <p className="text-sm font-bold" style={{ color: getRiskColor(data.riskValue) }}>
            Risk: {data.risk}%
          </p>
        </div>
      );
    }
    return null;
  };

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        No risk data available
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Stats Summary */}
      <div className="flex gap-4 text-xs">
        <div>
          <span className="text-gray-500">Current: </span>
          <span className="font-semibold" style={{ color: getRiskColor(chartData[chartData.length - 1]?.riskValue || 0) }}>
            {chartData[chartData.length - 1]?.risk}%
          </span>
        </div>
        <div>
          <span className="text-gray-500">Avg: </span>
          <span className="font-semibold">{averageRisk.toFixed(3)}%</span>
        </div>
        <div>
          <span className="text-gray-500">Peak: </span>
          <span className="font-semibold text-red-600">{maxRisk.toFixed(3)}%</span>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />}
          
          {/* Risk threshold lines */}
          <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" strokeOpacity={0.3} />
          <ReferenceLine y={60} stroke="#f59e0b" strokeDasharray="3 3" strokeOpacity={0.3} />
          <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.3} />
          
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
            domain={[0, 'auto']}
            label={{ value: 'Risk %', angle: -90, position: 'insideLeft', style: { fontSize: 10, fill: '#6b7280' } }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line 
            type="monotone" 
            dataKey="riskValue" 
            stroke="#3b82f6" 
            strokeWidth={2}
            dot={{ r: 2, fill: '#3b82f6' }}
            activeDot={{ r: 4 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

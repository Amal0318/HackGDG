import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

interface RiskTrendChartProps {
  data: Array<{ timestamp: string; risk_score: number }>;
}

export const RiskTrendChart = ({ data }: RiskTrendChartProps) => {
  const chartData = data.map((item, index) => ({
    index,
    risk: item.risk_score * 100,
    timestamp: new Date(item.timestamp).toLocaleTimeString(),
  }));

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
      <h3 className="text-gray-900 font-semibold mb-4">Risk Score Trend</h3>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="timestamp" 
            stroke="#6b7280"
            tick={{ fontSize: 12 }}
            interval="preserveStartEnd"
          />
          <YAxis 
            stroke="#6b7280"
            tick={{ fontSize: 12 }}
            domain={[0, 100]}
            label={{ value: 'Risk %', angle: -90, position: 'insideLeft', style: { fill: '#6b7280' } }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#ffffff',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
            }}
            labelStyle={{ color: '#1f2937' }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, 'Risk']}
          />
          
          {/* Threshold lines */}
          <ReferenceLine y={40} stroke="#eab308" strokeDasharray="3 3" label={{ value: 'Moderate', fill: '#eab308', fontSize: 10 }} />
          <ReferenceLine y={70} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: 'High', fill: '#f59e0b', fontSize: 10 }} />
          <ReferenceLine y={85} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Critical', fill: '#ef4444', fontSize: 10 }} />
          
          <Line
            type="monotone"
            dataKey="risk"
            stroke="#8b5cf6"
            strokeWidth={3}
            dot={{ fill: '#8b5cf6', r: 3 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>

      <div className="flex items-center justify-around mt-4 text-xs">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-primary-500" />
          <span className="text-gray-600">Low (&lt;40%)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <span className="text-gray-600">Moderate (40-70%)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-orange-500" />
          <span className="text-gray-600">High (70-85%)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span className="text-gray-600">Critical (&gt;85%)</span>
        </div>
      </div>
    </div>
  );
};

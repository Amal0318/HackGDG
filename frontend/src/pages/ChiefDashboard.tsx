import { useState } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Users, AlertCircle, BedDouble, Minus } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import {
  BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line
} from 'recharts';
import clsx from 'clsx';
import { getSystemStats, getAllAlerts, generateAlertHistory, generate7DayTrend, getFloorStats, getPatientsByFloor } from '../mockData';
import PatientDetailDrawer from '../components/PatientDetailDrawer';
import { patients } from '../mockData';

export default function ChiefDashboard() {
  const [selectedPatient, setSelectedPatient] = useState<typeof patients[0] | null>(null);
  const [selectedFloorDetail, setSelectedFloorDetail] = useState<number | null>(null);

  const stats = getSystemStats();
  const allAlerts = getAllAlerts();
  const criticalAlerts = allAlerts
    .filter(a => a.severity === 'critical')
    .slice(0, 10);
  const alertHistory = generateAlertHistory();

  const overviewCards = [
    {
      title: 'Total Patients',
      value: stats.totalPatients,
      trend: 'up' as const,
      trendValue: '+2',
      sparklineData: generate7DayTrend(stats.totalPatients),
      icon: Users,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50'
    },
    {
      title: 'Critical Alerts',
      value: stats.criticalAlerts,
      trend: 'down' as const,
      trendValue: '-1',
      sparklineData: generate7DayTrend(stats.criticalAlerts),
      icon: AlertCircle,
      color: 'text-red-600',
      bgColor: 'bg-red-50'
    },
    {
      title: 'Avg Risk Score',
      value: stats.avgRiskScore,
      trend: 'down' as const,
      trendValue: '↓',
      sparklineData: generate7DayTrend(stats.avgRiskScore),
      icon: TrendingDown,
      color: 'text-green-600',
      bgColor: 'bg-green-50'
    },
    {
      title: 'Bed Occupancy',
      value: `${stats.bedOccupancy}%`,
      trend: 'stable' as const,
      trendValue: '→',
      sparklineData: generate7DayTrend(stats.bedOccupancy),
      icon: BedDouble,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50'
    }
  ];

  const riskDistributionData = [
    { name: 'Low', value: stats.riskDistribution.low, color: '#10B981' },
    { name: 'Medium', value: stats.riskDistribution.medium, color: '#F59E0B' },
    { name: 'High', value: stats.riskDistribution.high, color: '#F97316' },
    { name: 'Critical', value: stats.riskDistribution.critical, color: '#EF4444' }
  ];

  const CountUp = ({ end }: { end: number }) => {
    const [count, setCount] = useState(0);

    useState(() => {
      const duration = 1000;
      const steps = 60;
      const increment = end / steps;
      let current = 0;

      const timer = setInterval(() => {
        current += increment;
        if (current >= end) {
          setCount(end);
          clearInterval(timer);
        } else {
          setCount(Math.floor(current));
        }
      }, duration / steps);

      return () => clearInterval(timer);
    });

    return <span>{count}</span>;
  };

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {overviewCards.map((card, index) => {
          const Icon = card.icon;
          const TrendIcon = card.trend === 'up' ? TrendingUp : card.trend === 'down' ? TrendingDown : Minus;
          
          return (
            <motion.div
              key={card.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className="bg-white rounded-lg shadow-md p-6"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <p className="text-xs text-gray-500 uppercase font-medium">{card.title}</p>
                  <p className="text-3xl font-bold text-gray-900 mt-2 font-mono">
                    {typeof card.value === 'number' ? <CountUp end={card.value} /> : card.value}
                  </p>
                  <div className="flex items-center gap-1 mt-2">
                    <TrendIcon className={clsx('w-4 h-4', 
                      card.trend === 'up' ? 'text-red-500' :
                      card.trend === 'down' ? 'text-green-500' :
                      'text-gray-400'
                    )} />
                    <span className="text-xs text-gray-600">{card.trendValue} from yesterday</span>
                  </div>
                </div>
                <div className={clsx('p-3 rounded-lg', card.bgColor)}>
                  <Icon className={clsx('w-6 h-6', card.color)} />
                </div>
              </div>
              
              {/* Mini Sparkline */}
              <div className="mt-4 h-12">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={card.sparklineData.map((v, i) => ({ value: v, index: i }))}>
                    <Line type="monotone" dataKey="value" stroke={card.color.replace('text-', '#')} strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Risk Heatmap */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-lg font-bold text-gray-900 mb-4">Floor Risk Heatmap</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[1, 2, 3].map((floor) => {
            const floorPatients = getPatientsByFloor(floor);
            const floorStats = getFloorStats(floor);
            
            return (
              <div key={floor} className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-gray-900">Floor {floor}</h3>
                  <button
                    onClick={() => setSelectedFloorDetail(floor)}
                    className="text-xs text-primary hover:text-primary-dark font-medium"
                  >
                    View Details
                  </button>
                </div>
                
                {/* 4x4 Grid */}
                <div className="grid grid-cols-4 gap-2">
                  {Array.from({ length: 16 }, (_, i) => {
                    const bedNum = i + 1;
                    const patient = floorPatients.find(p => {
                      const bedNumber = parseInt(p.bed.split('-')[1]);
                      return bedNumber === bedNum;
                    });
                    
                    return (
                      <div
                        key={bedNum}
                        onClick={() => patient && setSelectedPatient(patient)}
                        className={clsx(
                          'aspect-square rounded-md flex items-center justify-center text-sm font-bold cursor-pointer transition-transform hover:scale-110 relative group',
                          !patient && 'bg-gray-100 text-gray-400',
                          patient && patient.riskScore <= 30 && 'bg-green-200 text-green-800',
                          patient && patient.riskScore > 30 && patient.riskScore <= 50 && 'bg-yellow-200 text-yellow-800',
                          patient && patient.riskScore > 50 && patient.riskScore <= 70 && 'bg-orange-300 text-orange-900',
                          patient && patient.riskScore > 70 && 'bg-red-400 text-red-900'
                        )}
                      >
                        {bedNum}
                        {patient && (
                          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10 pointer-events-none">
                            {patient.name}<br />
                            Risk: {patient.riskScore}
                            {patient.alerts.length > 0 && <br />}
                            {patient.alerts.length > 0 && `${patient.alerts.length} alerts`}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
                
                {/* Floor Stats */}
                <div className="bg-gray-50 rounded p-3 text-xs space-y-1">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Total Beds:</span>
                    <span className="font-semibold">{floorStats.total}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">High Risk:</span>
                    <span className="font-semibold text-red-600">{floorStats.highRisk}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Avg Score:</span>
                    <span className="font-semibold">{floorStats.avgRiskScore}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Analytics Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Alerts by Hour */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">Alerts by Hour (Last 24h)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={alertHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" label={{ value: 'Hour', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'Alerts', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="critical" stackId="a" fill="#EF4444" name="Critical" />
              <Bar dataKey="warning" stackId="a" fill="#F59E0B" name="Warning" />
              <Bar dataKey="info" stackId="a" fill="#3B82F6" name="Info" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskDistributionData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {riskDistributionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-4 text-center">
            <p className="text-3xl font-bold text-gray-900">{stats.totalPatients}</p>
            <p className="text-sm text-gray-600">Total Patients</p>
          </div>
        </div>
      </div>

      {/* Critical Alerts Feed - Could be a fixed sidebar on larger screens */}
      <div className="bg-white rounded-lg shadow-md">
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-lg font-bold text-gray-900">Critical Alerts Feed</h3>
        </div>
        <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
          {criticalAlerts.length > 0 ? (
            criticalAlerts.map((alert: any, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="p-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
                      <span className="font-semibold text-sm text-gray-900 truncate">
                        {alert.patientName} • Bed {alert.patientBed}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700">{alert.message}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {formatDistanceToNow(alert.timestamp, { addSuffix: true })}
                    </p>
                  </div>
                  <button className="text-xs text-primary hover:text-primary-dark font-medium whitespace-nowrap flex-shrink-0">
                    View
                  </button>
                </div>
              </motion.div>
            ))
          ) : (
            <div className="p-8 text-center">
              <p className="text-gray-500">No critical alerts</p>
            </div>
          )}
        </div>
        {criticalAlerts.length > 0 && (
          <div className="p-4 border-t border-gray-200 text-center">
            <button className="text-sm text-primary hover:text-primary-dark font-medium">
              View All Alerts →
            </button>
          </div>
        )}
      </div>

      {/* Patient Detail Drawer */}
      <PatientDetailDrawer
        isOpen={selectedPatient !== null}
        onClose={() => setSelectedPatient(null)}
        patient={selectedPatient}
      />
    </div>
  );
}

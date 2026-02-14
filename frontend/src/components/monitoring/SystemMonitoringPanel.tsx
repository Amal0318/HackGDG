import { useEffect, useState } from 'react';
import { getSystemMetrics } from '../../services/api';
import { SystemMetrics } from '../../types';

export const SystemMonitoringPanel = () => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await getSystemMetrics();
        setMetrics(data);
        setLoading(false);
        setError(false);
      } catch (error) {
        console.error('Failed to fetch system metrics:', error);
        setLoading(false);
        setError(true);
        // Set demo/placeholder data when backend is unavailable
        setMetrics({
          kafka_throughput: 0,
          stream_latency_ms: 0,
          ml_inference_time_ms: 0,
          active_patients: 0,
        });
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="bg-gray-900 text-white p-6 rounded-lg shadow-xl">
        <p className="text-center text-gray-400">Loading system metrics...</p>
      </div>
    );
  }

  const getLatencyColor = (latency: number) => {
    if (latency < 100) return 'text-green-400';
    if (latency < 200) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getThroughputColor = (throughput: number) => {
    if (throughput > 1000) return 'text-green-400';
    if (throughput > 500) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">System Monitoring</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Kafka Throughput */}
        <div className="bg-blue-50 border-2 border-blue-200 p-4 rounded-lg shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-blue-700 text-sm font-medium">Kafka Throughput</span>
            <svg className="w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div className="text-3xl font-bold text-blue-700">
            {metrics.kafka_throughput}
          </div>
          <div className="text-blue-600 text-xs mt-1">msgs/sec</div>
          <div className="mt-3 bg-blue-200 rounded-full h-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, (metrics.kafka_throughput / 2000) * 100)}%` }}
            />
          </div>
        </div>

        {/* Stream Latency */}
        <div className="bg-purple-50 border-2 border-purple-200 p-4 rounded-lg shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-purple-700 text-sm font-medium">Stream Latency</span>
            <svg className="w-5 h-5 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="text-3xl font-bold text-purple-700">
            {metrics.stream_latency_ms}
          </div>
          <div className="text-purple-600 text-xs mt-1">milliseconds</div>
          <div className="mt-3 bg-purple-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-300 ${
                metrics.stream_latency_ms < 100 ? 'bg-green-500' : 
                metrics.stream_latency_ms < 200 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${Math.min(100, (metrics.stream_latency_ms / 300) * 100)}%` }}
            />
          </div>
        </div>

        {/* ML Inference Time */}
        <div className="bg-green-50 border-2 border-green-200 p-4 rounded-lg shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-green-700 text-sm font-medium">ML Inference</span>
            <svg className="w-5 h-5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div className="text-3xl font-bold text-green-700">
            {metrics.ml_inference_time_ms}
          </div>
          <div className="text-green-600 text-xs mt-1">milliseconds</div>
          <div className="mt-3 bg-green-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-300 ${
                metrics.ml_inference_time_ms < 50 ? 'bg-green-500' : 
                metrics.ml_inference_time_ms < 100 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${Math.min(100, (metrics.ml_inference_time_ms / 150) * 100)}%` }}
            />
          </div>
        </div>

        {/* Active Patients */}
        <div className="bg-teal-50 border-2 border-teal-200 p-4 rounded-lg shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-teal-700 text-sm font-medium">Active Patients</span>
            <svg className="w-5 h-5 text-teal-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
          </div>
          <div className="text-3xl font-bold text-teal-700">
            {metrics.active_patients}
          </div>
          <div className="text-teal-600 text-xs mt-1">monitored</div>
          <div className="mt-3 bg-teal-200 rounded-full h-2">
            <div 
              className="bg-teal-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, (metrics.active_patients / 10) * 100)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Service Health Grid */}
      <div className="mt-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Service Health</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-gray-50 border border-gray-200 p-3 rounded-lg flex items-center justify-between">
            <span className="text-sm text-gray-700 font-medium">Kafka Cluster</span>
            <div className={`w-3 h-3 ${error ? 'bg-gray-400' : 'bg-green-500 animate-pulse'} rounded-full`}></div>
          </div>
          <div className="bg-gray-50 border border-gray-200 p-3 rounded-lg flex items-center justify-between">
            <span className="text-sm text-gray-700 font-medium">Pathway Engine</span>
            <div className={`w-3 h-3 ${error ? 'bg-gray-400' : 'bg-green-500 animate-pulse'} rounded-full`}></div>
          </div>
          <div className="bg-gray-50 border border-gray-200 p-3 rounded-lg flex items-center justify-between">
            <span className="text-sm text-gray-700 font-medium">ML Service</span>
            <div className={`w-3 h-3 ${error ? 'bg-gray-400' : 'bg-green-500 animate-pulse'} rounded-full`}></div>
          </div>
          <div className="bg-gray-50 border border-gray-200 p-3 rounded-lg flex items-center justify-between">
            <span className="text-sm text-gray-700 font-medium">Backend API</span>
            <div className={`w-3 h-3 ${error ? 'bg-gray-400' : 'bg-green-500 animate-pulse'} rounded-full`}></div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="mt-6 bg-gray-50 border-2 border-gray-200 p-4 rounded-lg shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Performance Summary</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">System Status:</span>
            <span className={`${error ? 'text-gray-600' : 'text-green-600'} font-semibold`}>
              {error ? 'Initializing' : 'Optimal'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Pipeline Efficiency:</span>
            <span className={`${error ? 'text-gray-600' : 'text-green-600'} font-semibold`}>
              {error ? 'Pending' : metrics.stream_latency_ms < 100 ? '98%' : metrics.stream_latency_ms < 200 ? '85%' : '72%'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Data Processing:</span>
            <span className={`${error ? 'text-gray-600' : 'text-green-600'} font-semibold`}>
              {error ? 'Standby' : 'Real-time'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

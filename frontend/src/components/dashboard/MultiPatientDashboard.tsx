import { useState } from 'react';
import { useStore } from '../../store';
import { PatientGridCard } from './PatientGridCard';
import { PriorityQueue } from './PriorityQueue';
import { DashboardStats } from './DashboardStats';
import { ComparativeChart } from './ComparativeChart';

export const MultiPatientDashboard = () => {
  const patients = useStore(state => state.patients);
  const setSelectedPatient = useStore(state => state.setSelectedPatient);
  const selectedPatientId = useStore(state => state.selectedPatientId);
  
  const [compareMode, setCompareMode] = useState(false);
  const [comparedPatients, setComparedPatients] = useState<string[]>([]);
  const [compareMetric, setCompareMetric] = useState<'heart_rate' | 'spo2' | 'systolic_bp' | 'risk_score'>('heart_rate');
  
  // Get patient list
  const patientList = Array.from(patients.entries()).filter(([_, data]) => data.latest !== null);
  
  const handlePatientSelect = (patientId: string) => {
    if (compareMode) {
      // Toggle patient in comparison
      if (comparedPatients.includes(patientId)) {
        setComparedPatients(comparedPatients.filter(id => id !== patientId));
      } else if (comparedPatients.length < 4) {
        setComparedPatients([...comparedPatients, patientId]);
      }
    } else {
      // Navigate to single patient view
      setSelectedPatient(patientId);
      window.dispatchEvent(new CustomEvent('navigate-to-patient', { detail: { patientId } }));
    }
  };
  
  const handlePrioritySelect = (patientId: string) => {
    setSelectedPatient(patientId);
    window.dispatchEvent(new CustomEvent('navigate-to-patient', { detail: { patientId } }));
  };
  
  return (
    <main className="flex-1 bg-background-light p-6 overflow-y-auto">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-gray-900 text-3xl font-bold">Multi-Patient Dashboard</h1>
              <p className="text-gray-600 text-sm mt-1">
                Monitoring {patientList.length} patients in real-time
              </p>
            </div>
            
            {/* Compare Mode Toggle */}
            <div className="flex items-center space-x-3">
              <button
                onClick={() => {
                  setCompareMode(!compareMode);
                  if (compareMode) setComparedPatients([]);
                }}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  compareMode 
                    ? 'bg-primary-500 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {compareMode ? '✓ Compare Mode' : '📊 Compare Patients'}
              </button>
              
              {compareMode && comparedPatients.length > 0 && (
                <span className="bg-blue-100 text-blue-700 px-3 py-1 rounded-full text-sm font-medium">
                  {comparedPatients.length} selected
                </span>
              )}
            </div>
          </div>
        </div>
        
        {/* Dashboard Statistics */}
        <div className="mb-6">
          <DashboardStats />
        </div>
        
        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Priority Queue */}
          <div className="lg:col-span-1">
            <PriorityQueue onPatientSelect={handlePrioritySelect} />
          </div>
          
          {/* Patient Grid */}
          <div className="lg:col-span-3">
            {compareMode && comparedPatients.length >= 2 && (
              <div className="mb-6 space-y-4">
                {/* Metric Selector */}
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <p className="text-gray-700 text-sm font-medium mb-2">Compare Metric:</p>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { value: 'heart_rate', label: 'Heart Rate' },
                      { value: 'spo2', label: 'SpO2' },
                      { value: 'systolic_bp', label: 'Systolic BP' },
                      { value: 'risk_score', label: 'Risk Score' }
                    ].map(({ value, label }) => (
                      <button
                        key={value}
                        onClick={() => setCompareMetric(value as any)}
                        className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                          compareMetric === value
                            ? 'bg-primary-500 text-white'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
                
                {/* Comparative Chart */}
                <ComparativeChart 
                  patientIds={comparedPatients} 
                  metric={compareMetric}
                />
              </div>
            )}
            
            {/* Patient Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {patientList.map(([patientId, data]) => (
                <PatientGridCard
                  key={patientId}
                  patientId={patientId}
                  latestVitals={data.latest}
                  onSelect={() => handlePatientSelect(patientId)}
                  isSelected={
                    compareMode 
                      ? comparedPatients.includes(patientId)
                      : selectedPatientId === patientId
                  }
                />
              ))}
            </div>
            
            {patientList.length === 0 && (
              <div className="bg-white border border-gray-200 rounded-lg p-16 text-center">
                <svg className="mx-auto h-16 w-16 text-gray-300 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
                <p className="text-gray-400 text-lg">No active patients</p>
                <p className="text-gray-300 text-sm mt-2">Waiting for patient data stream...</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
};

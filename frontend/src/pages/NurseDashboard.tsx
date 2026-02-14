import { useState } from 'react';
import { Grid3X3, List, Grid, FileText } from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import { patients, getPatientsByWard } from '../mockData';
import PatientCard from '../components/PatientCard';
import PatientDetailDrawer from '../components/PatientDetailDrawer';
import ShiftHandoffModal from '../components/ShiftHandoffModal';

type ViewMode = 'grid' | 'list' | 'heatmap';
type FilterType = 'all' | 'high-risk' | 'active-alerts' | 'assigned';
type SortType = 'risk' | 'bed' | 'updated';

export default function NurseDashboard() {
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [filter, setFilter] = useState<FilterType>('all');
  const [sortBy, setSortBy] = useState<SortType>('risk');
  const [selectedPatient, setSelectedPatient] = useState<typeof patients[0] | null>(null);
  const [showHandoff, setShowHandoff] = useState(false);
  const [selectedFloor] = useState(3);
  const [selectedWard] = useState('A');

  const wardPatients = getPatientsByWard(selectedFloor, selectedWard);
  
  // Apply filters
  let filteredPatients = wardPatients;
  if (filter === 'high-risk') {
    filteredPatients = wardPatients.filter(p => p.riskLevel === 'high' || p.riskLevel === 'critical');
  } else if (filter === 'active-alerts') {
    filteredPatients = wardPatients.filter(p => p.alerts.length > 0);
  } else if (filter === 'assigned') {
    filteredPatients = wardPatients.filter(p => p.assignedNurse === 'Nurse Williams');
  }

  // Apply sorting
  const sortedPatients = [...filteredPatients].sort((a, b) => {
    if (sortBy === 'risk') return b.riskScore - a.riskScore;
    if (sortBy === 'bed') return a.bed.localeCompare(b.bed);
    return 0;
  });

  const occupancy = Math.round((wardPatients.length / 16) * 100);

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Floor {selectedFloor} — Ward {selectedWard}</h1>
            <p className="text-sm text-gray-600 mt-1">Day Shift (06:00–18:00)</p>
          </div>
          <div className="flex items-center gap-4">
            <div>
              <p className="text-xs text-gray-500 uppercase font-medium">Bed Occupancy</p>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex-1 w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-500"
                    style={{ width: `${occupancy}%` }}
                  />
                </div>
                <span className="text-sm font-semibold text-gray-900">{wardPatients.length}/16</span>
              </div>
            </div>
            <button
              onClick={() => setShowHandoff(true)}
              className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-dark transition-colors focus-visible-ring"
            >
              <FileText className="w-4 h-4" />
              <span className="hidden md:inline">Shift Handoff</span>
            </button>
          </div>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          {/* View Toggle */}
          <div className="flex items-center gap-2 bg-gray-100 p-1 rounded-lg">
            <button
              onClick={() => setViewMode('grid')}
              className={clsx(
                'flex items-center gap-2 px-3 py-2 rounded-md transition-colors focus-visible-ring',
                viewMode === 'grid' ? 'bg-white shadow-sm text-primary' : 'text-gray-600 hover:text-gray-900'
              )}
            >
              <Grid3X3 className="w-4 h-4" />
              <span className="text-sm font-medium hidden sm:inline">Grid</span>
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={clsx(
                'flex items-center gap-2 px-3 py-2 rounded-md transition-colors focus-visible-ring',
                viewMode === 'list' ? 'bg-white shadow-sm text-primary' : 'text-gray-600 hover:text-gray-900'
              )}
            >
              <List className="w-4 h-4" />
              <span className="text-sm font-medium hidden sm:inline">List</span>
            </button>
            <button
              onClick={() => setViewMode('heatmap')}
              className={clsx(
                'flex items-center gap-2 px-3 py-2 rounded-md transition-colors focus-visible-ring',
                viewMode === 'heatmap' ? 'bg-white shadow-sm text-primary' : 'text-gray-600 hover:text-gray-900'
              )}
            >
              <Grid className="w-4 h-4" />
              <span className="text-sm font-medium hidden sm:inline">Heatmap</span>
            </button>
          </div>

          {/* Filter Pills */}
          <div className="flex flex-wrap items-center gap-2">
            {(['all', 'high-risk', 'active-alerts', 'assigned'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={clsx(
                  'px-3 py-1.5 rounded-full text-sm font-medium transition-colors focus-visible-ring',
                  filter === f
                    ? 'bg-primary text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                )}
              >
                {f === 'all' && 'All'}
                {f === 'high-risk' && 'High Risk'}
                {f === 'active-alerts' && 'Active Alerts'}
                {f === 'assigned' && 'My Assigned'}
              </button>
            ))}
          </div>

          {/* Sort Dropdown */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortType)}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus-visible-ring bg-white"
          >
            <option value="risk">Risk ↓</option>
            <option value="bed">Bed #</option>
            <option value="updated">Last Updated</option>
          </select>
        </div>
      </div>

      {/* Grid View */}
      {viewMode === 'grid' && (
        <AnimatePresence mode="popLayout">
          <motion.div
            layout
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
          >
            {sortedPatients.map((patient, index) => (
              <PatientCard
                key={patient.id}
                patient={patient}
                onClick={() => setSelectedPatient(patient)}
                isAssigned={patient.assignedNurse === 'Nurse Williams'}
                index={index}
              />
            ))}
          </motion.div>
        </AnimatePresence>
      )}

      {/* List View */}
      {viewMode === 'list' && (
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Bed</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Name</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider hidden md:table-cell">HR</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider hidden md:table-cell">BP</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">SpO2</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider hidden lg:table-cell">RR</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider hidden lg:table-cell">Temp</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Risk</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Alerts</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {sortedPatients.map((patient) => (
                  <tr
                    key={patient.id}
                    className="hover:bg-gray-50 transition-colors cursor-pointer"
                    onClick={() => setSelectedPatient(patient)}
                  >
                    <td className="px-4 py-3 text-sm font-semibold text-gray-900">{patient.bed}</td>
                    <td className="px-4 py-3 text-sm text-gray-900">{patient.name}</td>
                    <td className="px-4 py-3 text-sm font-mono text-gray-700 hidden md:table-cell">{patient.vitals.heartRate}</td>
                    <td className="px-4 py-3 text-sm font-mono text-gray-700 hidden md:table-cell">
                      {patient.vitals.systolicBP}/{patient.vitals.diastolicBP}
                    </td>
                    <td className="px-4 py-3 text-sm font-mono text-gray-700">{patient.vitals.spo2}%</td>
                    <td className="px-4 py-3 text-sm font-mono text-gray-700 hidden lg:table-cell">{patient.vitals.respiratoryRate}</td>
                    <td className="px-4 py-3 text-sm font-mono text-gray-700 hidden lg:table-cell">{patient.vitals.temperature}°C</td>
                    <td className="px-4 py-3">
                      <span
                        className={clsx(
                          'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                          patient.riskLevel === 'low' && 'bg-green-100 text-green-800',
                          patient.riskLevel === 'medium' && 'bg-yellow-100 text-yellow-800',
                          patient.riskLevel === 'high' && 'bg-amber-100 text-amber-800',
                          patient.riskLevel === 'critical' && 'bg-red-100 text-red-800'
                        )}
                      >
                        {patient.riskScore}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">{patient.alerts.length}</td>
                    <td className="px-4 py-3">
                      <button className="text-primary hover:text-primary-dark text-sm font-medium focus-visible-ring">
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Heatmap View */}
      {viewMode === 'heatmap' && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="grid grid-cols-4 gap-4">
            {Array.from({ length: 16 }, (_, i) => {
              const bedNum = i + 1;
              const bed = `${selectedFloor}${selectedWard}-${bedNum}`;
              const patient = wardPatients.find(p => p.bed === bed);
              
              return (
                <motion.div
                  key={bed}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.05 }}
                  onClick={() => patient && setSelectedPatient(patient)}
                  className={clsx(
                    'aspect-square rounded-lg flex flex-col items-center justify-center cursor-pointer transition-transform hover:scale-105 relative group',
                    !patient && 'bg-gray-100 border-2 border-dashed border-gray-300',
                    patient && patient.riskScore <= 30 && 'bg-green-100 border-2 border-green-300',
                    patient && patient.riskScore > 30 && patient.riskScore <= 50 && 'bg-yellow-100 border-2 border-yellow-300',
                    patient && patient.riskScore > 50 && patient.riskScore <= 70 && 'bg-amber-200 border-2 border-amber-400',
                    patient && patient.riskScore > 70 && 'bg-red-300 border-2 border-red-500'
                  )}
                >
                  <span className="text-2xl font-bold font-mono">{bedNum}</span>
                  {patient && (
                    <div className="absolute inset-0 bg-black bg-opacity-90 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity p-3 flex flex-col justify-center">
                      <p className="text-white font-semibold text-sm">{patient.name}</p>
                      <div className="text-white text-xs mt-2 space-y-1">
                        <p>HR: {patient.vitals.heartRate}</p>
                        <p>SpO2: {patient.vitals.spo2}%</p>
                        <p>Risk: {patient.riskScore}</p>
                      </div>
                    </div>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>
      )}

      {/* Patient Detail Drawer */}
      <PatientDetailDrawer
        isOpen={selectedPatient !== null}
        onClose={() => setSelectedPatient(null)}
        patient={selectedPatient}
      />

      {/* Shift Handoff Modal */}
      <ShiftHandoffModal
        isOpen={showHandoff}
        onClose={() => setShowHandoff(false)}
        patients={wardPatients}
        nurseName="Nurse Williams"
        shift="day"
        floor={selectedFloor}
        ward={selectedWard}
      />
    </div>
  );
}

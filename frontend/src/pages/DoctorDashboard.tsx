import { useState } from 'react';
import { Search } from 'lucide-react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { patients, getPatientsByFloor, getPatientsByDoctor, getFloorStats } from '../mockData';
import PatientCard from '../components/PatientCard';
import PatientDetailDrawer from '../components/PatientDetailDrawer';
import RiskBadge from '../components/RiskBadge';

type SortType = 'risk' | 'floor' | 'name';

export default function DoctorDashboard() {
  const [selectedFloor, setSelectedFloor] = useState<number | 'all'>('all');
  const [selectedPatient, setSelectedPatient] = useState<typeof patients[0] | null>(null);
  const [sortBy, setSortBy] = useState<SortType>('risk');
  const [searchQuery, setSearchQuery] = useState('');

  const doctorName = 'Dr. Anderson';
  const myPatients = getPatientsByDoctor(doctorName);

  const displayPatients = selectedFloor === 'all' ? patients : getPatientsByFloor(selectedFloor);
  
  // Apply sorting
  const sortedPatients = [...displayPatients].sort((a, b) => {
    if (sortBy === 'risk') return b.riskScore - a.riskScore;
    if (sortBy === 'floor') return a.floor - b.floor;
    if (sortBy === 'name') return a.name.localeCompare(b.name);
    return 0;
  });

  // Filter by search
  const filteredPatients = searchQuery
    ? sortedPatients.filter(p => 
        p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.bed.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : sortedPatients;

  const floorStats = [1, 2, 3].map(floor => ({
    floor,
    ...getFloorStats(floor)
  }));

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-full">
      {/* Left Sidebar - My Patients */}
      <div className="lg:w-80 flex-shrink-0">
        <div className="bg-white rounded-lg shadow-md p-4 sticky top-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-gray-900">My Patients</h2>
            <span className="bg-primary text-white text-sm font-semibold px-2 py-1 rounded-full">
              {myPatients.length}
            </span>
          </div>

          <div className="mb-4">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as SortType)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus-visible-ring bg-white"
            >
              <option value="risk">Sort by Risk</option>
              <option value="floor">Sort by Floor</option>
              <option value="name">Sort Alphabetically</option>
            </select>
          </div>

          <div className="space-y-2 max-h-[calc(100vh-300px)] overflow-y-auto">
            {myPatients
              .sort((a, b) => {
                if (sortBy === 'risk') return b.riskScore - a.riskScore;
                if (sortBy === 'floor') return a.floor - b.floor;
                return a.name.localeCompare(b.name);
              })
              .map((patient) => (
                <motion.div
                  key={patient.id}
                  whileHover={{ scale: 1.02 }}
                  onClick={() => {
                    setSelectedPatient(patient);
                    setSelectedFloor(patient.floor);
                  }}
                  className="bg-gray-50 hover:bg-primary-light rounded-lg p-3 cursor-pointer transition-colors border border-gray-200"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-sm text-gray-900 truncate">{patient.name}</p>
                      <p className="text-xs text-gray-600 mt-1">Bed {patient.bed}</p>
                      <p className="text-xs text-gray-500">Floor {patient.floor}</p>
                    </div>
                    <RiskBadge score={patient.riskScore} size="sm" showLabel={false} />
                  </div>
                  <button className="mt-2 text-xs text-primary hover:text-primary-dark font-medium">
                    View Details →
                  </button>
                </motion.div>
              ))}
          </div>
        </div>
      </div>

      {/* Main Area */}
      <div className="flex-1 min-w-0 space-y-6">
        {/* Search and Floor Tabs */}
        <div className="bg-white rounded-lg shadow-md p-4">
          <div className="flex flex-col md:flex-row md:items-center gap-4">
            {/* Search Bar */}
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search patient name or bed number..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus-visible-ring"
              />
            </div>
          </div>

          {/* Floor Tabs */}
          <div className="flex items-center gap-2 mt-4">
            <button
              onClick={() => setSelectedFloor('all')}
              className={clsx(
                'px-4 py-2 rounded-md text-sm font-medium transition-colors focus-visible-ring',
                selectedFloor === 'all'
                  ? 'bg-primary text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              )}
            >
              All Floors
            </button>
            {[1, 2, 3].map((floor) => (
              <button
                key={floor}
                onClick={() => setSelectedFloor(floor)}
                className={clsx(
                  'px-4 py-2 rounded-md text-sm font-medium transition-colors focus-visible-ring',
                  selectedFloor === floor
                    ? 'bg-primary text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                )}
              >
                Floor {floor}
              </button>
            ))}
          </div>
        </div>

        {/* Floor Summary Cards */}
        {selectedFloor !== 'all' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {floorStats
              .filter(stats => stats.floor === selectedFloor)
              .map((stats) => (
                <div key={stats.floor} className="space-y-4">
                  <div className="bg-white rounded-lg shadow-md p-4">
                    <p className="text-xs text-gray-500 uppercase font-medium">Beds Occupied</p>
                    <p className="text-2xl font-bold text-gray-900 mt-1">{stats.total}</p>
                  </div>
                  <div className="bg-white rounded-lg shadow-md p-4">
                    <p className="text-xs text-gray-500 uppercase font-medium">High Risk</p>
                    <p className="text-2xl font-bold text-red-600 mt-1">{stats.highRisk}</p>
                  </div>
                  <div className="bg-white rounded-lg shadow-md p-4">
                    <p className="text-xs text-gray-500 uppercase font-medium">Active Alerts</p>
                    <p className="text-2xl font-bold text-amber-600 mt-1">{stats.activeAlerts}</p>
                  </div>
                </div>
              ))}
          </div>
        )}

        {/* Patient Grid */}
        <motion.div
          layout
          className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4"
        >
          {filteredPatients.length > 0 ? (
            filteredPatients.map((patient, index) => (
              <PatientCard
                key={patient.id}
                patient={patient}
                onClick={() => setSelectedPatient(patient)}
                isAssigned={patient.assignedDoctor === doctorName}
                index={index}
              />
            ))
          ) : (
            <div className="col-span-full text-center py-12">
              <p className="text-lg text-gray-500">No patients found</p>
              <p className="text-sm text-gray-400 mt-1">Try adjusting your search or filters</p>
            </div>
          )}
        </motion.div>
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

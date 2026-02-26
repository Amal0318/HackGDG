import { useState, useMemo, useEffect, useRef } from 'react';
import { Search, Activity } from 'lucide-react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { usePatients, Patient } from '../hooks/usePatients';
import PatientCard from '../components/PatientCard';
import PatientDetailDrawer from '../components/PatientDetailDrawer';
import RiskBadge from '../components/RiskBadge';
import { usePatientRiskHistory } from '../hooks/usePatientRiskHistory';

type SortType = 'risk' | 'floor' | 'name';

export default function DoctorDashboard() {
  const [selectedFloor, setSelectedFloor] = useState<number | 'all'>('all');
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [sortBy, setSortBy] = useState<SortType>('risk');
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch real-time patient data from backend
  const { patients, error } = usePatients({ refreshInterval: 5000 });
  
  // Connect to WebSocket for live updates
  const { isLive, subscribeToFloorHistory } = usePatientRiskHistory();
  
  // Subscribe to all floors for live updates
  useMemo(() => {
    if (isLive) {
      ['1F', '2F', '3F'].forEach(floor => subscribeToFloorHistory(floor));
    }
  }, [isLive, subscribeToFloorHistory]);
  
  // Filter patients by floor if selected
  const displayPatients = selectedFloor === 'all' 
    ? patients 
    : patients.filter(p => p.floor === selectedFloor);
  
  // Apply sorting
  const sortedPatients = useMemo(() => {
    return [...displayPatients].sort((a, b) => {
      if (sortBy === 'risk') return b.riskScore - a.riskScore;
      if (sortBy === 'floor') return a.floor - b.floor;
      if (sortBy === 'name') return a.name.localeCompare(b.name);
      return 0;
    });
  }, [displayPatients, sortBy]);

  // Filter by search
  const filteredPatients = useMemo(() => {
    return searchQuery
      ? sortedPatients.filter(p => 
          p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          p.bed.toLowerCase().includes(searchQuery.toLowerCase())
        )
      : sortedPatients;
  }, [sortedPatients, searchQuery]);

  // Stable display order – only re-orders when user explicitly changes sort/search
  const [displayOrder, setDisplayOrder] = useState<string[]>([]);
  const isFirstLoad = useRef(true);

  useEffect(() => {
    if (filteredPatients.length > 0 && isFirstLoad.current) {
      isFirstLoad.current = false;
      setDisplayOrder(filteredPatients.map(p => p.patient_id));
    }
  }, [filteredPatients]);

  const handleSortChange = (newSort: SortType) => {
    setSortBy(newSort);
    const reordered = [...filteredPatients]
      .sort((a, b) => {
        if (newSort === 'risk') return b.riskScore - a.riskScore;
        if (newSort === 'floor') return a.floor - b.floor;
        if (newSort === 'name') return a.name.localeCompare(b.name);
        return 0;
      })
      .map(p => p.patient_id);
    setDisplayOrder(reordered);
  };

  useEffect(() => {
    if (filteredPatients.length > 0) {
      setDisplayOrder(prev => {
        // Add any new patients to the end
        const existing = new Set(prev);
        const newIds = filteredPatients
          .filter(p => !existing.has(p.patient_id))
          .map(p => p.patient_id);
        // Remove patients that are no longer in the filtered list
        const currentIds = new Set(filteredPatients.map(p => p.patient_id));
        const filtered = prev.filter(id => currentIds.has(id));
        return newIds.length > 0 ? [...filtered, ...newIds] : filtered;
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchQuery, selectedFloor]);

  // Apply stable order – vitals update in-place, no shuffling on live updates
  const stablePatients = useMemo(() => {
    if (displayOrder.length === 0) return filteredPatients;
    const orderMap = new Map(displayOrder.map((id, i) => [id, i]));
    return [...filteredPatients].sort((a, b) => {
      const aIdx = orderMap.get(a.patient_id) ?? 999;
      const bIdx = orderMap.get(b.patient_id) ?? 999;
      return aIdx - bIdx;
    });
  }, [filteredPatients, displayOrder]);

  // Calculate floor stats
  const floorStats = useMemo(() => {
    return [1, 2, 3].map(floor => {
      const floorPatients = patients.filter(p => p.floor === floor);
      const highRisk = floorPatients.filter(p => p.riskScore > 70).length;
      const activeAlerts = floorPatients.filter(p => p.alerts && p.alerts.length > 0).length;
      
      return {
        floor,
        total: floorPatients.length,
        highRisk,
        activeAlerts,
      };
    });
  }, [patients]);

  // All patients assigned to this doctor
  const myPatients = patients;
  
  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <p className="text-lg text-red-600 font-semibold">Error loading patients</p>
          <p className="text-sm text-gray-600 mt-2">{error}</p>
          <p className="text-xs text-gray-500 mt-4">Make sure the backend API is running on http://localhost:8000</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-full">
      {/* Left Sidebar - My Patients */}
      <div className="lg:w-80 flex-shrink-0">
        <div className="bg-white rounded-lg shadow-md p-4 sticky top-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-gray-900">My Patients</h2>
            <div className="flex items-center gap-2">
              {isLive && (
                <span className="flex items-center gap-1 text-xs text-green-600 font-medium">
                  <Activity className="w-3 h-3 animate-pulse" />
                  Live
                </span>
              )}
              <span className="bg-primary text-white text-sm font-semibold px-2 py-1 rounded-full">
                {myPatients.length}
              </span>
            </div>
          </div>

          <div className="mb-4">
            <select
              value={sortBy}
              onChange={(e) => handleSortChange(e.target.value as SortType)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus-visible-ring bg-white"
            >
              <option value="risk">Sort by Risk</option>
              <option value="floor">Sort by Floor</option>
              <option value="name">Sort Alphabetically</option>
            </select>
          </div>

          <p className="text-xs text-gray-400 mb-2">{myPatients.length} patients total</p>
          <div className="space-y-2 max-h-[calc(100vh-230px)] overflow-y-auto pr-1">
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
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {stablePatients.length > 0 ? (
            stablePatients.map((patient) => (
              <PatientCard
                key={patient.patient_id}
                patient={patient}
                onClick={() => setSelectedPatient(patient)}
              />
            ))
          ) : (
            <div className="col-span-full text-center py-12">
              <p className="text-lg text-gray-500">No patients found</p>
              <p className="text-sm text-gray-400 mt-1">Try adjusting your search or filters</p>
            </div>
          )}
        </div>
      </div>

      {/* Patient Detail Drawer */}
      {selectedPatient && (
        <PatientDetailDrawer
          onClose={() => setSelectedPatient(null)}
          patient={selectedPatient}
        />
      )}
    </div>
  );
}

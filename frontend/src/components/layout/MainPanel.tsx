import { useState } from 'react';
import { useStore } from '../../store';
import { VitalCard } from '../vitals/VitalCard';
import { TrendChart } from '../vitals/TrendChart';
import { StateBadge, EventBadge } from '../vitals/Badges';
import { RiskScoreCard } from '../risk/RiskScoreCard';
import { RiskTrendChart } from '../risk/RiskTrendChart';
import { AlertBanner } from '../risk/AlertBanner';
import { AnomalyIndicator } from '../anomaly/AnomalyIndicator';
import { EventTimeline } from '../anomaly/EventTimeline';
import { AnomalyDetailsModal } from '../anomaly/AnomalyDetailsModal';
import { AlertAcknowledgment } from '../alerts/AlertAcknowledgment';
import { DataExport } from '../export/DataExport';
import { EventFilter, FilterOptions } from '../filters/EventFilter';
import { VitalMessage } from '../../types';
import { PhaseInfo } from '../info/PhaseInfo';

const getVitalStatus = (value: number, normal: [number, number]): 'normal' | 'warning' | 'critical' => {
  if (value < normal[0] * 0.8 || value > normal[1] * 1.2) return 'critical';
  if (value < normal[0] || value > normal[1]) return 'warning';
  return 'normal';
};

export const MainPanel = () => {
  const selectedPatientId = useStore(state => state.selectedPatientId);
  const patients = useStore(state => state.patients);
  const acknowledgedAlerts = useStore(state => state.acknowledgedAlerts);
  const acknowledgeAlert = useStore(state => state.acknowledgeAlert);
  const clearAcknowledgment = useStore(state => state.clearAcknowledgment);
  const setSelectedEvent = useStore(state => state.setSelectedEvent);
  const selectedEvent = useStore(state => state.selectedEvent);
  const selectedPhase = useStore(state => state.selectedPhase);
  
  // Phase 4 local state
  const [filters, setFilters] = useState<FilterOptions>({
    eventTypes: ['NONE', 'HYPOTENSION', 'TACHYCARDIA', 'HYPOXIA', 'SEPSIS_ALERT'],
    states: ['STABLE', 'EARLY_DETERIORATION', 'CRITICAL', 'INTERVENTION'],
    showAnomalies: true,
    dateRange: 'all'
  });

  if (!selectedPatientId) {
    return (
      <main className="flex-1 bg-background-light p-6 flex items-center justify-center">
        <div className="text-gray-500 text-center">
          <p className="text-xl">No patient selected</p>
          <p className="text-sm mt-2">Select a patient from the sidebar to view vitals</p>
        </div>
      </main>
    );
  }

  const patientData = patients.get(selectedPatientId);
  
  if (!patientData?.latest) {
    return (
      <main className="flex-1 bg-background-light p-6 flex items-center justify-center">
        <div className="text-gray-500 text-center">
          <p className="text-xl">Waiting for data...</p>
          <p className="text-sm mt-2">Patient: {selectedPatientId}</p>
        </div>
      </main>
    );
  }

  const { latest, history, riskHistory } = patientData;

  // Extract trend data
  const hrData = history.map(h => ({ timestamp: h.timestamp, value: h.heart_rate }));
  const spo2Data = history.map(h => ({ timestamp: h.timestamp, value: h.spo2 }));
  const shockIndexData = history.map(h => ({ timestamp: h.timestamp, value: h.shock_index }));
  const systolicBpData = history.map(h => ({ timestamp: h.timestamp, value: h.systolic_bp }));

  // Check if risk score is available
  const hasRiskScore = latest.risk_score !== undefined;
  const hasAnomaly = latest.anomaly_detected || false;
  const anomalyType = latest.anomaly_type;
  
  // Phase 4 - Alert acknowledgment status
  const isAlertAcknowledged = acknowledgedAlerts.has(selectedPatientId);
  
  // Phase 4 - Filter events
  const filteredHistory = history.filter(event => {
    // Filter by event type
    if (!filters.eventTypes.includes(event.event_type)) return false;
    
    // Filter by state
    if (!filters.states.includes(event.state)) return false;
    
    // Filter by anomaly
    if (!filters.showAnomalies && event.anomaly_detected) return false;
    
    // Filter by date range
    if (filters.dateRange !== 'all') {
      const eventTime = new Date(event.timestamp).getTime();
      const now = Date.now();
      const hourAgo = now - (60 * 60 * 1000);
      const dayAgo = now - (24 * 60 * 60 * 1000);
      
      if (filters.dateRange === 'last-hour' && eventTime < hourAgo) return false;
      if (filters.dateRange === 'last-day' && eventTime < dayAgo) return false;
    }
    
    return true;
  });
  
  const handleAcknowledge = (note: string) => {
    if (isAlertAcknowledged) {
      clearAcknowledgment(selectedPatientId);
    } else {
      acknowledgeAlert(selectedPatientId, note);
    }
  };

  return (
    <main className="flex-1 bg-background-light p-6 overflow-y-auto">
      <div className="max-w-7xl mx-auto">
        {/* Phase Info Card */}
        <PhaseInfo />

        {/* Patient Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-gray-900 text-3xl font-bold">{selectedPatientId}</h2>
              <p className="text-gray-600 text-sm mt-1">
                Last updated: {new Date(latest.timestamp).toLocaleString()}
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <StateBadge state={latest.state} />
              <EventBadge event={latest.event_type} />
              {/* Phase 3+ - Anomaly Indicator */}
              {(selectedPhase === 'phase3' || selectedPhase === 'phase4' || selectedPhase === 'all') && hasAnomaly && (
                <AnomalyIndicator hasAnomaly={hasAnomaly} anomalyType={anomalyType} />
              )}
            </div>
          </div>
        </div>

        {/* Phase 2+ - Alert Banner */}
        {(selectedPhase === 'phase2' || selectedPhase === 'phase3' || selectedPhase === 'phase4' || selectedPhase === 'all') && hasRiskScore && latest.risk_score !== undefined && (
          <div className="space-y-4 mb-6">
            <AlertBanner riskScore={latest.risk_score} patientId={selectedPatientId} />
            {/* Phase 4+ - Alert Acknowledgment */}
            {(selectedPhase === 'phase4' || selectedPhase === 'all') && (latest.risk_score >= 50 || hasAnomaly) && (
              <AlertAcknowledgment
                patientId={selectedPatientId}
                riskScore={latest.risk_score}
                onAcknowledge={handleAcknowledge}
                isAcknowledged={isAlertAcknowledged}
              />
            )}
          </div>
        )}

        {/* Vital Cards Grid - All Phases */}
        <div className={`grid grid-cols-1 md:grid-cols-2 ${(selectedPhase === 'phase2' || selectedPhase === 'all') && hasRiskScore ? 'lg:grid-cols-4' : 'lg:grid-cols-3'} gap-4 mb-6`}>
          <VitalCard
            title="Heart Rate"
            value={latest.heart_rate}
            unit="bpm"
            status={getVitalStatus(latest.heart_rate, [60, 100])}
          />
          <VitalCard
            title="Systolic BP"
            value={latest.systolic_bp}
            unit="mmHg"
            status={getVitalStatus(latest.systolic_bp, [90, 120])}
          />
          <VitalCard
            title="Diastolic BP"
            value={latest.diastolic_bp}
            unit="mmHg"
            status={getVitalStatus(latest.diastolic_bp, [60, 80])}
          />
          <VitalCard
            title="SpO2"
            value={latest.spo2}
            unit="%"
            status={getVitalStatus(latest.spo2, [95, 100])}
          />
          <VitalCard
            title="Respiratory Rate"
            value={latest.respiratory_rate}
            unit="bpm"
            status={getVitalStatus(latest.respiratory_rate, [12, 20])}
          />
          <VitalCard
            title="Temperature"
            value={latest.temperature}
            unit="°C"
            status={getVitalStatus(latest.temperature, [36.5, 37.5])}
          />
          <VitalCard
            title="Shock Index"
            value={latest.shock_index}
            unit=""
            status={getVitalStatus(latest.shock_index, [0.5, 0.7])}
          />
          
          {/* Risk Score Card - Phase 2+ */}
          {(selectedPhase === 'phase2' || selectedPhase === 'phase3' || selectedPhase === 'phase4' || selectedPhase === 'all') && hasRiskScore && latest.risk_score !== undefined && (
            <div className="md:col-span-2 lg:col-span-1">
              <RiskScoreCard 
                riskScore={latest.risk_score} 
                riskLevel={latest.risk_level}
              />
            </div>
          )}
        </div>

        {/* Trend Charts Grid - All Phases */}
        {history.length > 1 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <TrendChart
              title="Heart Rate Trend"
              data={hrData}
              color="#ef4444"
              domain={[40, 140]}
            />
            <TrendChart
              title="SpO2 Trend"
              data={spo2Data}
              color="#3b82f6"
              domain={[90, 100]}
            />
            <TrendChart
              title="Systolic BP Trend"
              data={systolicBpData}
              color="#10b981"
              domain={[60, 180]}
            />
            <TrendChart
              title="Shock Index Trend"
              data={shockIndexData}
              color="#f59e0b"
              domain={[0, 2]}
            />
          </div>
        )}

        {/* Risk Trend Chart - Phase 2+ */}
        {(selectedPhase === 'phase2' || selectedPhase === 'phase3' || selectedPhase === 'phase4' || selectedPhase === 'all') && hasRiskScore && riskHistory.length > 1 && (
          <RiskTrendChart data={riskHistory} />
        )}

        {/* Phase 4 - Advanced Features Grid */}
        {(selectedPhase === 'phase4' || selectedPhase === 'all') && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
            {/* Event Filter - Phase 4 */}
            <div className="lg:col-span-1">
              <EventFilter onFilterChange={setFilters} />
            </div>
            
            {/* Data Export - Phase 4 */}
            <div className="lg:col-span-1">
              <DataExport patientId={selectedPatientId} vitalHistory={history} />
            </div>
          </div>
        )}

        {/* Event Timeline - Phase 3+ with Phase 4 enhancements */}
        {(selectedPhase === 'phase3' || selectedPhase === 'phase4' || selectedPhase === 'all') && filteredHistory.length > 0 && (
          <EventTimeline 
            events={filteredHistory} 
            onEventClick={(event) => setSelectedEvent(event)}
          />
        )}
        
        {/* Anomaly Details Modal - Phase 4 */}
        {(selectedPhase === 'phase4' || selectedPhase === 'all') && (
          <AnomalyDetailsModal
            event={selectedEvent}
            isOpen={selectedEvent !== null}
            onClose={() => setSelectedEvent(null)}
          />
        )}
      </div>
    </main>
  );
};

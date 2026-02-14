import { VitalMessage } from '../../types';

interface DataExportProps {
  patientId: string;
  vitalHistory: VitalMessage[];
}

export const DataExport = ({ patientId, vitalHistory }: DataExportProps) => {
  const exportToCSV = () => {
    if (!vitalHistory.length) {
      alert('No data to export');
      return;
    }

    // CSV headers
    const headers = [
      'Timestamp',
      'Patient ID',
      'Heart Rate',
      'Systolic BP',
      'Diastolic BP',
      'SpO2',
      'Respiratory Rate',
      'Temperature',
      'Shock Index',
      'State',
      'Event Type',
      'Risk Score',
      'Risk Level',
      'Anomaly Detected',
      'Anomaly Type'
    ];

    // Convert data to CSV rows
    const rows = vitalHistory.map(vital => [
      vital.timestamp,
      vital.patient_id,
      vital.heart_rate,
      vital.systolic_bp,
      vital.diastolic_bp,
      vital.spo2,
      vital.respiratory_rate,
      vital.temperature,
      vital.shock_index,
      vital.state,
      vital.event_type,
      vital.risk_score ?? '',
      vital.risk_level ?? '',
      vital.anomaly_detected ?? '',
      vital.anomaly_type ?? ''
    ]);

    // Combine headers and rows
    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    // Create and download file
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', `${patientId}_vitals_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportToJSON = () => {
    if (!vitalHistory.length) {
      alert('No data to export');
      return;
    }

    const exportData = {
      patient_id: patientId,
      export_timestamp: new Date().toISOString(),
      record_count: vitalHistory.length,
      vitals: vitalHistory
    };

    const jsonContent = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', `${patientId}_vitals_${new Date().toISOString().split('T')[0]}.json`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getDataSummary = () => {
    if (!vitalHistory.length) return 'No data available';
    
    const oldest = new Date(vitalHistory[0].timestamp);
    const newest = new Date(vitalHistory[vitalHistory.length - 1].timestamp);
    const duration = Math.round((newest.getTime() - oldest.getTime()) / 1000 / 60); // minutes
    
    return `${vitalHistory.length} records spanning ${duration} minutes`;
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <h3 className="text-gray-900 text-lg font-semibold mb-2">📥 Export Patient Data</h3>
      <p className="text-gray-600 text-sm mb-4">{getDataSummary()}</p>
      
      <div className="flex space-x-3">
        <button
          onClick={exportToCSV}
          disabled={!vitalHistory.length}
          className="flex-1 bg-primary-500 hover:bg-primary-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
        >
          <span>📊</span>
          <span>Export CSV</span>
        </button>
        <button
          onClick={exportToJSON}
          disabled={!vitalHistory.length}
          className="flex-1 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
        >
          <span>📄</span>
          <span>Export JSON</span>
        </button>
      </div>
      
      <p className="text-gray-500 text-xs mt-3">
        Export includes all vitals, risk scores, and anomaly data for {patientId}
      </p>
    </div>
  );
};

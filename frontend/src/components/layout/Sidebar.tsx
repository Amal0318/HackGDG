import { useStore } from '../../store';
import { PatientState } from '../../types';

const getStateBgColor = (state: PatientState): string => {
  switch (state) {
    case 'STABLE':
      return 'bg-primary-500';
    case 'EARLY_DETERIORATION':
      return 'bg-yellow-500';
    case 'CRITICAL':
      return 'bg-red-500';
    case 'INTERVENTION':
      return 'bg-purple-500';
    default:
      return 'bg-gray-500';
  }
};

const getStateTextColor = (state: PatientState): string => {
  switch (state) {
    case 'STABLE':
      return 'text-primary-600';
    case 'EARLY_DETERIORATION':
      return 'text-yellow-600';
    case 'CRITICAL':
      return 'text-red-600';
    case 'INTERVENTION':
      return 'text-purple-600';
    default:
      return 'text-gray-600';
  }
};

export const Sidebar = () => {
  const patientList = useStore(state => state.patientList);
  const selectedPatientId = useStore(state => state.selectedPatientId);
  const setSelectedPatient = useStore(state => state.setSelectedPatient);
  const patients = useStore(state => state.patients);

  return (
    <aside className="bg-white border-r border-gray-200 w-64 overflow-y-auto">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-gray-900 font-semibold text-lg">Patients</h2>
        <p className="text-gray-600 text-sm">Active: {patientList.length}</p>
      </div>

      <div className="p-2">
        {patientList.map((patient) => {
          const patientData = patients.get(patient.patient_id);
          const currentState = patientData?.latest?.state || patient.state;
          const isSelected = selectedPatientId === patient.patient_id;

          return (
            <button
              key={patient.patient_id}
              onClick={() => setSelectedPatient(patient.patient_id)}
              className={`w-full text-left p-3 mb-2 rounded-lg transition-all ${
                isSelected
                  ? 'bg-primary-500 text-white shadow-lg'
                  : 'bg-gray-50 hover:bg-gray-100 text-gray-900'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`font-semibold ${isSelected ? 'text-white' : 'text-gray-900'}`}>{patient.patient_id}</span>
                <div className={`w-2 h-2 rounded-full ${getStateBgColor(currentState)}`}></div>
              </div>
              <div className="flex items-center space-x-1 mt-1">
                <span className={`text-xs font-medium ${isSelected ? 'text-white' : getStateTextColor(currentState)}`}>
                  {currentState.replace('_', ' ')}
                </span>
              </div>
            </button>
          );
        })}

        {patientList.length === 0 && (
          <div className="text-center text-gray-500 py-8">
            <p className="text-sm">No patients available</p>
          </div>
        )}
      </div>
    </aside>
  );
};

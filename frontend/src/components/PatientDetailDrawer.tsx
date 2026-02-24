import { Fragment } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { X, AlertTriangle, FileText } from 'lucide-react';
import RiskBadge from './RiskBadge';
import VitalSign from './VitalSign';
import TrendsView from './TrendsView';
import { Patient } from '../hooks/usePatients';
import { PDFGenerator } from '../utils/pdfGenerator';

interface PatientDetailDrawerProps {
  patient: Patient | null;
  onClose: () => void;
}

const PatientDetailDrawer: React.FC<PatientDetailDrawerProps> = ({ patient, onClose }) => {
  if (!patient) return null;

  const handleExportPDF = async () => {
    await PDFGenerator.generatePatientPDF(patient);
  };

  return (
    <Transition appear show={true} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black bg-opacity-25" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-300"
                enterFrom="translate-x-full"
                enterTo="translate-x-0"
                leave="transform transition ease-in-out duration-300"
                leaveFrom="translate-x-0"
                leaveTo="translate-x-full"
              >
                <Dialog.Panel className="pointer-events-auto w-screen max-w-full md:max-w-2xl">
                  <div className="flex h-full flex-col overflow-y-scroll bg-white shadow-xl">
                    {/* Header */}
                    <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-6">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-4">
                          <RiskBadge score={patient.latest_risk_score} size="lg" />
                          <div>
                            <Dialog.Title className="text-2xl font-bold text-white">
                              {patient.name}
                            </Dialog.Title>
                            <p className="text-blue-100">Bed {patient.bed_number} • Floor {patient.floor}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={handleExportPDF}
                            className="rounded-md p-2 text-white hover:bg-white/20 transition-colors flex items-center gap-2"
                            title="Export to PDF"
                          >
                            <FileText className="h-5 w-5" />
                            <span className="hidden md:inline text-sm">Export PDF</span>
                          </button>
                          <button
                            onClick={onClose}
                            className="rounded-md p-2 text-white hover:bg-white/20 transition-colors"
                          >
                            <X className="h-6 w-6" />
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Content */}
                    <div className="flex-1 px-6 py-6 space-y-6">
                      {/* Current Vitals */}
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Vitals</h3>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                          <VitalSign label="Heart Rate" value={patient.vitals.heart_rate} unit="bpm" />
                          <VitalSign label="Blood Pressure" value={`${patient.vitals.systolic_bp}/${patient.vitals.diastolic_bp}`} unit="mmHg" />
                          <VitalSign label="Respiratory Rate" value={patient.vitals.respiratory_rate} unit="br/min" />
                          <VitalSign label="Temperature" value={patient.vitals.temperature.toFixed(1)} unit="°C" />
                          <VitalSign label="SpO2" value={patient.vitals.spo2} unit="%" />
                        </div>
                      </div>

                      {/* Live Patient Trends - Vitals & Risk */}
                      <TrendsView 
                        patientId={patient.patient_id}
                        className=""
                      />

                      {/* Abnormal Vitals */}
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Abnormal Vitals</h3>
                        {patient.abnormal_vitals && patient.abnormal_vitals.length > 0 ? (
                          <div className="space-y-2">
                            {patient.abnormal_vitals.map((vital) => (
                              <div key={`${vital.vital}-${vital.value}`} className="flex items-center p-3 bg-red-50 border border-red-200 rounded-lg">
                                <AlertTriangle className="h-5 w-5 text-red-500 mr-3 flex-shrink-0" />
                                <div className="flex-1">
                                  <span className="font-semibold text-gray-900">{vital.vital}:</span>
                                  <span className="ml-2 text-red-600 font-bold">{vital.value} {vital.unit}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-gray-500 text-sm">No abnormal vitals detected.</p>
                        )}
                      </div>
                    </div>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
};

export default PatientDetailDrawer;

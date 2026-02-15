import { Fragment } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { X, Printer, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { format } from 'date-fns';
import { Patient as MockPatient } from '../mockData';
import RiskBadge from './RiskBadge';
import clsx from 'clsx';

interface ShiftHandoffModalProps {
  isOpen: boolean;
  onClose: () => void;
  patients: MockPatient[];
  nurseName: string;
  shift: 'day' | 'night';
  floor: number;
  ward: string;
}

export default function ShiftHandoffModal({
  isOpen,
  onClose,
  patients,
  nurseName,
  shift,
  floor,
  ward
}: ShiftHandoffModalProps) {
  const shiftTimes = {
    day: '06:00 – 18:00',
    night: '18:00 – 06:00'
  };

  const handlePrint = () => {
    window.print();
  };

  const getTrendIcon = (trend: 'improving' | 'stable' | 'deteriorating') => {
    switch (trend) {
      case 'improving':
        return <TrendingDown className="w-4 h-4 text-green-600" />;
      case 'deteriorating':
        return <TrendingUp className="w-4 h-4 text-red-600" />;
      default:
        return <Minus className="w-4 h-4 text-gray-400" />;
    }
  };

  return (
    <Transition appear show={isOpen} as={Fragment}>
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

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel className="w-full max-w-5xl transform overflow-hidden rounded-2xl bg-white shadow-xl transition-all">
                {/* Header */}
                <div className="bg-gradient-to-r from-primary to-primary-dark px-6 py-6 no-print">
                  <div className="flex items-start justify-between">
                    <div>
                      <Dialog.Title className="text-2xl font-bold text-white">
                        Shift Handoff Report
                      </Dialog.Title>
                      <p className="text-primary-light mt-1">
                        {format(new Date(), 'MMMM dd, yyyy • HH:mm')}
                      </p>
                    </div>
                    <button
                      onClick={onClose}
                      className="rounded-md p-2 text-white hover:bg-white/20 transition-colors"
                    >
                      <X className="h-6 w-6" />
                    </button>
                  </div>
                </div>

                {/* Printable Content */}
                <div className="p-6 max-h-[70vh] overflow-y-auto">
                  {/* Shift Summary */}
                  <div className="bg-gray-50 rounded-lg p-4 mb-6 print-break-inside-avoid">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <p className="text-xs text-gray-500 uppercase font-medium">Nurse</p>
                        <p className="font-semibold text-gray-900">{nurseName}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 uppercase font-medium">Shift</p>
                        <p className="font-semibold text-gray-900 capitalize">{shift} Shift</p>
                        <p className="text-xs text-gray-600">{shiftTimes[shift]}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 uppercase font-medium">Location</p>
                        <p className="font-semibold text-gray-900">Floor {floor} — Ward {ward}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 uppercase font-medium">Total Patients</p>
                        <p className="font-semibold text-gray-900">{patients.length}</p>
                      </div>
                    </div>
                  </div>

                  {/* Patient Summary Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {patients.map((patient) => {
                      const criticalVitals = [];
                      if (patient.vitals.heartRate > 120 || patient.vitals.heartRate < 60) {
                        criticalVitals.push(`HR: ${patient.vitals.heartRate} bpm`);
                      }
                      if (patient.vitals.spo2 < 92) {
                        criticalVitals.push(`SpO2: ${patient.vitals.spo2}%`);
                      }
                      if (patient.vitals.systolicBP < 90 || patient.vitals.systolicBP > 180) {
                        criticalVitals.push(`BP: ${patient.vitals.systolicBP}/${patient.vitals.diastolicBP}`);
                      }

                      return (
                        <div
                          key={patient.id}
                          className={clsx(
                            'bg-white border rounded-lg p-4 print-break-inside-avoid',
                            patient.riskLevel === 'critical' && 'border-red-300 bg-red-50',
                            patient.riskLevel === 'high' && 'border-amber-300 bg-amber-50',
                            patient.riskLevel === 'medium' && 'border-yellow-300',
                            patient.riskLevel === 'low' && 'border-gray-300'
                          )}
                        >
                          <div className="flex items-start justify-between mb-3">
                            <div className="flex-1">
                              <h3 className="font-bold text-lg text-gray-900">{patient.name}</h3>
                              <p className="text-sm text-gray-600">Bed {patient.bed} • {patient.age} years</p>
                              <p className="text-xs text-gray-500 mt-1">{patient.diagnosis}</p>
                            </div>
                            <RiskBadge score={patient.riskScore} size="sm" showLabel={false} />
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-sm">
                              <span className="text-gray-600">Risk Trend:</span>
                              <div className="flex items-center gap-1">
                                {getTrendIcon(patient.predictive.riskTrend)}
                                <span className={clsx(
                                  'font-semibold',
                                  patient.predictive.riskTrend === 'improving' && 'text-green-600',
                                  patient.predictive.riskTrend === 'deteriorating' && 'text-red-600',
                                  patient.predictive.riskTrend === 'stable' && 'text-gray-600'
                                )}>
                                  {patient.predictive.riskTrend}
                                </span>
                              </div>
                            </div>

                            {criticalVitals.length > 0 && (
                              <div>
                                <p className="text-xs text-gray-500 uppercase font-medium mb-1">Critical Vitals:</p>
                                <div className="flex flex-wrap gap-1">
                                  {criticalVitals.map((vital, idx) => (
                                    <span
                                      key={idx}
                                      className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full font-medium"
                                    >
                                      {vital}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}

                            <div className="grid grid-cols-2 gap-2 text-xs">
                              <div>
                                <span className="text-gray-500">HR:</span>{' '}
                                <span className="font-mono font-semibold">{patient.vitals.heartRate}</span>
                              </div>
                              <div>
                                <span className="text-gray-500">SpO2:</span>{' '}
                                <span className="font-mono font-semibold">{patient.vitals.spo2}%</span>
                              </div>
                              <div>
                                <span className="text-gray-500">BP:</span>{' '}
                                <span className="font-mono font-semibold">
                                  {patient.vitals.systolicBP}/{patient.vitals.diastolicBP}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-500">RR:</span>{' '}
                                <span className="font-mono font-semibold">{patient.vitals.respiratoryRate}</span>
                              </div>
                            </div>

                            {patient.interventions.length > 0 && (
                              <div>
                                <p className="text-xs text-gray-500 uppercase font-medium mb-1">Recent Interventions:</p>
                                <p className="text-xs text-gray-700">{patient.interventions.length} logged</p>
                              </div>
                            )}

                            {patient.alerts.length > 0 && (
                              <div className="bg-amber-100 border border-amber-300 rounded px-2 py-1">
                                <p className="text-xs font-semibold text-amber-900">
                                  {patient.alerts.filter(a => !a.acknowledged).length} Pending Alerts
                                </p>
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Notes Section */}
                  <div className="mt-6 border-t border-gray-200 pt-6 print-break-inside-avoid">
                    <h3 className="font-semibold text-gray-900 mb-2">Additional Notes</h3>
                    <div className="bg-gray-50 rounded-lg p-4 min-h-[100px]">
                      <p className="text-sm text-gray-500 italic">No additional notes for this shift.</p>
                    </div>
                  </div>
                </div>

                {/* Footer Actions */}
                <div className="bg-gray-50 px-6 py-4 flex items-center justify-between no-print border-t border-gray-200">
                  <p className="text-sm text-gray-600">
                    {patients.length} patients • {patients.filter(p => p.riskLevel === 'critical' || p.riskLevel === 'high').length} high risk
                  </p>
                  <div className="flex gap-2">
                    <button
                      onClick={handlePrint}
                      className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-dark transition-colors focus-visible-ring"
                    >
                      <Printer className="w-4 h-4" />
                      Print Report
                    </button>
                    <button
                      onClick={onClose}
                      className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors focus-visible-ring"
                    >
                      Close
                    </button>
                  </div>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}

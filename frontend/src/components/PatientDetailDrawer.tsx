import { Fragment, useState } from 'react';
import { Dialog, Transition, Tab } from '@headlessui/react';
import { X, Activity, TrendingUp, Bell, Clipboard, Check, AlertTriangle, Plus } from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, AreaChart } from 'recharts';
import clsx from 'clsx';
import { Patient } from '../mockData';
import VitalSign from './VitalSign';
import RiskBadge from './RiskBadge';
import { getVitalStatus } from '../utils/calculations';

interface PatientDetailDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  patient: Patient | null;
  onIntervention?: (type: string, details: string) => void;
}

export default function PatientDetailDrawer({ isOpen, onClose, patient, onIntervention }: PatientDetailDrawerProps) {
  const [selectedTab, setSelectedTab] = useState(0);

  if (!patient) return null;

  const trendChartData = patient.trends.heartRate.map((_, idx) => ({
    time: idx * 5, // 5-minute intervals
    heartRate: patient.trends.heartRate[idx],
    systolicBP: patient.trends.systolicBP[idx],
    spo2: patient.trends.spo2[idx],
    respiratoryRate: patient.trends.respiratoryRate[idx]
  }));

  // Add predictive data
  const predictiveData = patient.predictive.heartRate.map((hr, idx) => ({
    time: trendChartData.length * 5 + idx * 5,
    heartRate: hr,
    spo2: patient.predictive.spo2[idx],
    isPredicted: true
  }));

  const fullChartData = [...trendChartData, ...predictiveData];

  const handleQuickIntervention = (type: string) => {
    const details = {
      '500ml NS': '500ml Normal Saline bolus',
      '2L O2': 'Oxygen therapy at 2L/min',
      'PRN Med': 'PRN medication administered',
      'Custom': ''
    }[type] || '';

    if (type === 'Custom') {
      const customDetails = prompt('Enter intervention details:');
      if (customDetails) {
        onIntervention?.(type, customDetails);
      }
    } else if (details) {
      onIntervention?.(type, details);
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

        <div className="fixed inset-0 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-300"
                enterFrom="translate-x-full"
                enterTo="translateX-0"
                leave="transform transition ease-in-out duration-300"
                leaveFrom="translate-x-0"
                leaveTo="translate-x-full"
              >
                <Dialog.Panel className="pointer-events-auto w-screen max-w-full md:max-w-2xl lg:max-w-3xl">
                  <div className="flex h-full flex-col overflow-y-scroll bg-white shadow-xl">
                    {/* Header */}
                    <div className="bg-gradient-to-r from-primary to-primary-dark px-6 py-6">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-4">
                          <RiskBadge score={patient.riskScore} size="md" />
                          <div>
                            <Dialog.Title className="text-2xl font-bold text-white">
                              {patient.name}
                            </Dialog.Title>
                            <p className="text-primary-light">{patient.age} years • Bed {patient.bed}</p>
                            <p className="text-sm text-primary-light mt-1">
                              Admitted {format(patient.admissionDate, 'MMM dd, yyyy')}
                            </p>
                          </div>
                        </div>
                        <button
                          onClick={onClose}
                          className="rounded-md p-2 text-white hover:bg-white/20 transition-colors focus-visible-ring"
                        >
                          <X className="h-6 w-6" />
                        </button>
                      </div>
                    </div>

                    {/* Tabs */}
                    <Tab.Group selectedIndex={selectedTab} onChange={setSelectedTab}>
                      <Tab.List className="flex border-b border-gray-200 px-6 bg-gray-50">
                        {['Vitals', 'Trends', 'Alerts', 'Interventions'].map((tab, idx) => (
                          <Tab
                            key={tab}
                            className={({ selected }) =>
                              clsx(
                                'px-4 py-3 text-sm font-medium border-b-2 transition-colors focus-visible-ring',
                                selected
                                  ? 'border-primary text-primary'
                                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                              )
                            }
                          >
                            <div className="flex items-center gap-2">
                              {idx === 0 && <Activity className="w-4 h-4" />}
                              {idx === 1 && <TrendingUp className="w-4 h-4" />}
                              {idx === 2 && <Bell className="w-4 h-4" />}
                              {idx === 3 && <Clipboard className="w-4 h-4" />}
                              {tab}
                              {idx === 2 && patient.alerts.length > 0 && (
                                <span className="ml-1 bg-red-500 text-white text-xs px-2 py-0.5 rounded-full">
                                  {patient.alerts.length}
                                </span>
                              )}
                            </div>
                          </Tab>
                        ))}
                      </Tab.List>

                      <Tab.Panels className="flex-1 overflow-y-auto p-6">
                        {/* Vitals Tab */}
                        <Tab.Panel className="space-y-6">
                          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                            <p className="text-sm text-blue-800 font-medium">Diagnosis: {patient.diagnosis}</p>
                            <p className="text-xs text-blue-600 mt-1">Assigned to: {patient.assignedDoctor} • {patient.assignedNurse}</p>
                          </div>

                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                            <VitalSign
                              label="Heart Rate"
                              value={patient.vitals.heartRate}
                              unit="bpm"
                              status={getVitalStatus(patient.vitals.heartRate, [60, 100])}
                              trendData={patient.trends.heartRate}
                            />
                            <VitalSign
                              label="Systolic BP"
                              value={patient.vitals.systolicBP}
                              unit="mmHg"
                              status={getVitalStatus(patient.vitals.systolicBP, [90, 140])}
                              trendData={patient.trends.systolicBP}
                            />
                            <VitalSign
                              label="Diastolic BP"
                              value={patient.vitals.diastolicBP}
                              unit="mmHg"
                              status={getVitalStatus(patient.vitals.diastolicBP, [60, 90])}
                            />
                            <VitalSign
                              label="SpO2"
                              value={patient.vitals.spo2}
                              unit="%"
                              status={getVitalStatus(patient.vitals.spo2, [95, 100])}
                              trendData={patient.trends.spo2}
                            />
                            <VitalSign
                              label="Respiratory Rate"
                              value={patient.vitals.respiratoryRate}
                              unit="/min"
                              status={getVitalStatus(patient.vitals.respiratoryRate, [12, 20])}
                              trendData={patient.trends.respiratoryRate}
                            />
                            <VitalSign
                              label="Temperature"
                              value={patient.vitals.temperature}
                              unit="°C"
                              status={getVitalStatus(patient.vitals.temperature, [36.5, 37.5])}
                              trendData={patient.trends.temperature}
                            />
                            <VitalSign
                              label="GCS"
                              value={patient.vitals.gcs}
                              unit=""
                              status={patient.vitals.gcs === 15 ? 'normal' : patient.vitals.gcs >= 13 ? 'warning' : 'critical'}
                            />
                            <VitalSign
                              label="CVP"
                              value={patient.vitals.cvp}
                              unit="mmHg"
                              status="normal"
                            />
                            <VitalSign
                              label="MAP"
                              value={patient.vitals.map}
                              unit="mmHg"
                              status={getVitalStatus(patient.vitals.map, [65, 110])}
                            />
                            <VitalSign
                              label="Urine Output"
                              value={patient.vitals.urine}
                              unit="ml/hr"
                              status={getVitalStatus(patient.vitals.urine, [30, 100])}
                            />
                            <VitalSign
                              label="Lactate"
                              value={patient.vitals.lactate}
                              unit="mmol/L"
                              status={patient.vitals.lactate < 2 ? 'normal' : patient.vitals.lactate < 4 ? 'warning' : 'critical'}
                            />
                            <VitalSign
                              label="Glucose"
                              value={patient.vitals.glucose}
                              unit="mg/dL"
                              status={getVitalStatus(patient.vitals.glucose, [70, 180])}
                            />
                            <VitalSign
                              label="FiO2"
                              value={patient.vitals.fio2}
                              unit="%"
                              status="normal"
                            />
                            <VitalSign
                              label="PEEP"
                              value={patient.vitals.peep}
                              unit="cmH2O"
                              status="normal"
                            />
                          </div>

                          <p className="text-xs text-gray-500 text-center">
                            Last updated: {formatDistanceToNow(new Date(), { addSuffix: true })}
                          </p>
                        </Tab.Panel>

                        {/* Trends Tab */}
                        <Tab.Panel className="space-y-6">
                          <div className="bg-white rounded-lg border border-gray-200 p-4">
                            <h3 className="font-semibold text-gray-900 mb-4">4-Hour Trend with 20-Minute Prediction</h3>
                            <ResponsiveContainer width="100%" height={300}>
                              <AreaChart data={fullChartData}>
                                <defs>
                                  <linearGradient id="colorHR" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#EF4444" stopOpacity={0.8}/>
                                    <stop offset="95%" stopColor="#EF4444" stopOpacity={0}/>
                                  </linearGradient>
                                  <linearGradient id="colorSpO2" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                                  </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis 
                                  dataKey="time" 
                                  label={{ value: 'Time (minutes)', position: 'insideBottom', offset: -5 }}
                                />
                                <YAxis yAxisId="left" label={{ value: 'HR / Sys BP', angle: -90, position: 'insideLeft' }} />
                                <YAxis yAxisId="right" orientation="right" label={{ value: 'SpO2 / RR', angle: 90, position: 'insideRight' }} />
                                <Tooltip />
                                <Legend />
                                <ReferenceLine x={trendChartData.length * 5} stroke="#666" strokeDasharray="3 3" label="Now" />
                                <Area 
                                  yAxisId="left" 
                                  type="monotone" 
                                  dataKey="heartRate" 
                                  stroke="#EF4444" 
                                  strokeWidth={2}
                                  fill="url(#colorHR)"
                                  strokeDasharray={(d: any) => d.isPredicted ? "5 5" : "0"}
                                />
                                <Line 
                                  yAxisId="left" 
                                  type="monotone" 
                                  dataKey="systolicBP" 
                                  stroke="#F59E0B" 
                                  strokeWidth={2}
                                  strokeDasharray={(d: any) => d.isPredicted ? "5 5" : "0"}
                                />
                                <Area 
                                  yAxisId="right" 
                                  type="monotone" 
                                  dataKey="spo2" 
                                  stroke="#3B82F6" 
                                  strokeWidth={2}
                                  fill="url(#colorSpO2)"
                                  strokeDasharray={(d: any) => d.isPredicted ? "5 5" : "0"}
                                />
                                <Line 
                                  yAxisId="right" 
                                  type="monotone" 
                                  dataKey="respiratoryRate" 
                                  stroke="#10B981" 
                                  strokeWidth={2}
                                  strokeDasharray={(d: any) => d.isPredicted ? "5 5" : "0"}
                                />
                              </AreaChart>
                            </ResponsiveContainer>
                          </div>

                          {/* Predictive Alert */}
                          {patient.predictive.estimatedTimeToAlert && (
                            <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded-r-lg">
                              <div className="flex items-start gap-3">
                                <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5" />
                                <div className="flex-1">
                                  <h4 className="font-semibold text-amber-900">Predicted Deterioration</h4>
                                  <p className="text-sm text-amber-800 mt-1">
                                    Based on current trends, patient may require intervention in approximately {patient.predictive.estimatedTimeToAlert} minutes.
                                  </p>
                                  <div className="flex gap-2 mt-3">
                                    <button className="px-3 py-1 bg-amber-600 text-white text-sm rounded-md hover:bg-amber-700 transition-colors">
                                      Acknowledge
                                    </button>
                                    <button className="px-3 py-1 bg-red-600 text-white text-sm rounded-md hover:bg-red-700 transition-colors">
                                      Escalate Now
                                    </button>
                                    <button className="px-3 py-1 border border-amber-600 text-amber-700 text-sm rounded-md hover:bg-amber-50 transition-colors">
                                      Pre-intervene
                                    </button>
                                  </div>
                                </div>
                              </div>
                            </div>
                          )}
                        </Tab.Panel>

                        {/* Alerts Tab */}
                        <Tab.Panel>
                          {patient.alerts.length === 0 ? (
                            <div className="text-center py-12">
                              <Check className="w-16 h-16 text-green-500 mx-auto mb-4" />
                              <p className="text-lg font-semibold text-gray-700">No Active Alerts</p>
                              <p className="text-sm text-gray-500 mt-2">This patient is stable</p>
                            </div>
                          ) : (
                            <div className="space-y-3">
                              {patient.alerts
                                .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
                                .map((alert, idx) => (
                                  <div
                                    key={alert.id}
                                    className={clsx(
                                      'border-l-4 p-4 rounded-r-lg',
                                      alert.severity === 'critical' && 'bg-red-50 border-red-500',
                                      alert.severity === 'warning' && 'bg-amber-50 border-amber-500',
                                      alert.severity === 'info' && 'bg-blue-50 border-blue-500'
                                    )}
                                  >
                                    <div className="flex items-start justify-between">
                                      <div>
                                        <div className="flex items-center gap-2">
                                          <span
                                            className={clsx(
                                              'px-2 py-1 text-xs font-semibold rounded-full',
                                              alert.severity === 'critical' && 'bg-red-100 text-red-800',
                                              alert.severity === 'warning' && 'bg-amber-100 text-amber-800',
                                              alert.severity === 'info' && 'bg-blue-100 text-blue-800'
                                            )}
                                          >
                                            {alert.severity.toUpperCase()}
                                          </span>
                                          {alert.vital && (
                                            <span className="text-xs text-gray-600 font-medium">{alert.vital}</span>
                                          )}
                                        </div>
                                        <p className="mt-2 text-sm text-gray-900">{alert.message}</p>
                                        <p className="mt-1 text-xs text-gray-500">
                                          {formatDistanceToNow(alert.timestamp, { addSuffix: true })}
                                        </p>
                                      </div>
                                      {!alert.acknowledged && (
                                        <button className="px-3 py-1 text-xs font-medium bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors">
                                          Acknowledge
                                        </button>
                                      )}
                                    </div>
                                  </div>
                                ))}
                            </div>
                          )}
                        </Tab.Panel>

                        {/* Interventions Tab */}
                        <Tab.Panel className="space-y-6">
                          <div>
                            <h3 className="font-semibold text-gray-900 mb-3">Quick Log</h3>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                              {['500ml NS', '2L O2', 'PRN Med', 'Custom'].map((type) => (
                                <button
                                  key={type}
                                  onClick={() => handleQuickIntervention(type)}
                                  className="flex items-center justify-center gap-2 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-dark transition-colors focus-visible-ring"
                                >
                                  <Plus className="w-4 h-4" />
                                  {type}
                                </button>
                              ))}
                            </div>
                          </div>

                          <div>
                            <h3 className="font-semibold text-gray-900 mb-3">Timeline</h3>
                            <div className="relative">
                              <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-300"></div>
                              <div className="space-y-4">
                                {patient.interventions.map((intervention, idx) => (
                                  <div key={intervention.id} className="relative pl-10">
                                    <div className="absolute left-2.5 w-3 h-3 bg-primary rounded-full border-2 border-white"></div>
                                    <div className="bg-gray-50 rounded-lg p-3">
                                      <div className="flex items-start justify-between">
                                        <div>
                                          <p className="font-semibold text-sm text-gray-900">{intervention.type}</p>
                                          <p className="text-sm text-gray-600 mt-1">{intervention.details}</p>
                                          <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
                                            <span>{intervention.user}</span>
                                            <span>•</span>
                                            <span>{format(intervention.timestamp, 'MMM dd, HH:mm')}</span>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        </Tab.Panel>
                      </Tab.Panels>
                    </Tab.Group>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}

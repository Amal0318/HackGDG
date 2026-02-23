import React, { useState } from 'react';
import { X, Play, RotateCcw, Activity, AlertTriangle, TrendingDown, Heart } from 'lucide-react';

interface DeveloperToolsModalProps {
  isOpen: boolean;
  onClose: () => void;
  patients: { id: string; name: string }[];
}

// Move scenarios outside component to prevent recreation on every render
const scenarios = [
  {
    id: 'sepsis',
    name: 'Sepsis Episode',
    description: 'Tachycardia, hypotension, elevated lactate & WBC',
    icon: Activity,
    color: 'bg-orange-500',
    defaultSeverity: 0.8,
    defaultDuration: 300,
  },
  {
    id: 'shock',
    name: 'Septic Shock',
    description: 'Severe tachycardia, severe hypotension, critical',
    icon: AlertTriangle,
    color: 'bg-red-600',
    defaultSeverity: 0.9,
    defaultDuration: 240,
  },
  {
    id: 'mild_deterioration',
    name: 'Mild Deterioration',
    description: 'Subtle changes for early warning demo',
    icon: TrendingDown,
    color: 'bg-yellow-500',
    defaultSeverity: 0.3,
    defaultDuration: 180,
  },
  {
    id: 'critical',
    name: 'Critical Condition',
    description: 'Immediately set to critical state',
    icon: Heart,
    color: 'bg-red-700',
    defaultSeverity: 1.0,
    defaultDuration: 600,
  },
  {
    id: 'recovery',
    name: 'Rapid Recovery',
    description: 'Returns patient to stable condition',
    icon: RotateCcw,
    color: 'bg-green-600',
    defaultSeverity: 0,
    defaultDuration: 0,
  },
] as const;

export function DeveloperToolsModal({ isOpen, onClose, patients }: DeveloperToolsModalProps) {
  const [selectedPatient, setSelectedPatient] = useState<string>('');
  const [selectedScenario, setSelectedScenario] = useState<string>('sepsis');
  const [severity, setSeverity] = useState<number>(0.8);
  const [duration, setDuration] = useState<number>(300);
  const [isTriggering, setIsTriggering] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);

  if (!isOpen) return null;

  // Debug log
  console.log('🎮 DeveloperToolsModal IS RENDERING!');
  console.log('DeveloperToolsModal rendering with patients:', patients);
  console.log('isOpen:', isOpen);

  const currentScenario = scenarios.find(s => s.id === selectedScenario);

  const handleTriggerScenario = async () => {
    if (!selectedPatient) {
      setMessage({ type: 'error', text: 'Please select a patient' });
      return;
    }

    setIsTriggering(true);
    setMessage(null);

    try {
      const response = await fetch('http://localhost:5001/api/dev/scenarios/trigger', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          patient_id: selectedPatient,
          scenario_type: selectedScenario,
          severity: severity,
          duration: duration,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessage({
          type: 'success',
          text: `✅ ${currentScenario?.name} triggered for ${selectedPatient}`,
        });
      } else {
        setMessage({
          type: 'error',
          text: `❌ Error: ${data.error || 'Failed to trigger scenario'}`,
        });
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text: `❌ Connection error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    } finally {
      setIsTriggering(false);
    }
  };

  const handleResetPatient = async () => {
    if (!selectedPatient) {
      setMessage({ type: 'error', text: 'Please select a patient' });
      return;
    }

    setIsTriggering(true);
    setMessage(null);

    try {
      const response = await fetch('http://localhost:5001/api/dev/scenarios/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          patient_id: selectedPatient,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessage({
          type: 'success',
          text: `✅ ${selectedPatient} reset to healthy baseline`,
        });
      } else {
        setMessage({
          type: 'error',
          text: `❌ Error: ${data.error || 'Failed to reset patient'}`,
        });
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text: `❌ Connection error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    } finally {
      setIsTriggering(false);
    }
  };

  // Update severity and duration when scenario changes
  React.useEffect(() => {
    const scenario = scenarios.find(s => s.id === selectedScenario);
    if (scenario) {
      setSeverity(scenario.defaultSeverity);
      setDuration(scenario.defaultDuration);
    }
  }, [selectedScenario]);

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-gray-900 border-4 border-red-500 rounded-lg shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        {/* TEST BANNER */}
        <div className="bg-yellow-400 text-black p-4 text-center font-bold text-xl">
          🎮 DEVELOPER TOOLS MODAL - IF YOU SEE THIS, IT'S RENDERING! 🎮
        </div>
        
        {/* Header */}
        <div className="sticky top-0 bg-gray-900 border-b border-gray-700 p-6 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-3">
              <span className="text-3xl">🎮</span>
              Developer Tools
            </h2>
            <p className="text-gray-400 text-sm mt-1">
              Trigger clinical scenarios for presentations and demos
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
          >
            <X className="w-6 h-6 text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Patient Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Select Patient
            </label>
            <select
              value={selectedPatient}
              onChange={(e) => setSelectedPatient(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">
                {!patients || patients.length === 0 ? 'Loading patients...' : 'Choose a patient...'}
              </option>
              {patients && patients.map((patient) => (
                <option key={patient.id} value={patient.id}>
                  {patient.id} - {patient.name}
                </option>
              ))}
            </select>
          </div>

          {/* Scenario Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-3">
              Select Scenario
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {scenarios.map((scenario) => {
                const Icon = scenario.icon;
                return (
                  <button
                    key={scenario.id}
                    onClick={() => setSelectedScenario(scenario.id)}
                    className={`p-4 rounded-lg border-2 transition-all text-left ${
                      selectedScenario === scenario.id
                        ? 'border-blue-500 bg-blue-500/10'
                        : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`${scenario.color} p-2 rounded-lg`}>
                        <Icon className="w-5 h-5 text-white" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-semibold text-white">
                          {scenario.name}
                        </h3>
                        <p className="text-xs text-gray-400 mt-1">
                          {scenario.description}
                        </p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Parameters */}
          {selectedScenario !== 'recovery' && selectedScenario !== 'critical' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Severity */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Severity: {severity.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={severity}
                  onChange={(e) => setSeverity(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Mild</span>
                  <span>Moderate</span>
                  <span>Severe</span>
                </div>
              </div>

              {/* Duration */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Duration: {Math.floor(duration / 60)}m {duration % 60}s
                </label>
                <input
                  type="range"
                  min="60"
                  max="600"
                  step="30"
                  value={duration}
                  onChange={(e) => setDuration(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>1min</span>
                  <span>5min</span>
                  <span>10min</span>
                </div>
              </div>
            </div>
          )}

          {/* Message Display */}
          {message && (
            <div
              className={`p-4 rounded-lg ${
                message.type === 'success'
                  ? 'bg-green-500/10 border border-green-500/50 text-green-400'
                  : 'bg-red-500/10 border border-red-500/50 text-red-400'
              }`}
            >
              {message.text}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4">
            <button
              onClick={handleTriggerScenario}
              disabled={isTriggering || !selectedPatient}
              className="flex-1 py-3 px-6 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-semibold rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
              <Play className="w-5 h-5" />
              {isTriggering ? 'Triggering...' : 'Trigger Scenario'}
            </button>

            <button
              onClick={handleResetPatient}
              disabled={isTriggering || !selectedPatient}
              className="py-3 px-6 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-white font-semibold rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
              <RotateCcw className="w-5 h-5" />
              Reset to Baseline
            </button>
          </div>

          {/* Quick Help */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-white mb-2">💡 Quick Tips</h3>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>• Scenarios trigger gradual deterioration over time (not instant)</li>
              <li>• Watch the ML risk scores climb as vitals worsen</li>
              <li>• Use "Reset to Baseline" to return patient to healthy state</li>
              <li>• "Critical Condition" sets extreme values immediately for high-risk demos</li>
              <li>• Changes are visible in the dashboard graph within 30-60 seconds</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

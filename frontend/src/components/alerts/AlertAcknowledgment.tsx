import { useState } from 'react';

interface AlertAcknowledgmentProps {
  patientId: string;
  riskScore: number;
  onAcknowledge: (note: string) => void;
  isAcknowledged: boolean;
}

export const AlertAcknowledgment = ({ 
  patientId, 
  riskScore, 
  onAcknowledge,
  isAcknowledged 
}: AlertAcknowledgmentProps) => {
  const [showNoteInput, setShowNoteInput] = useState(false);
  const [note, setNote] = useState('');

  const handleAcknowledge = () => {
    if (showNoteInput && note.trim()) {
      onAcknowledge(note);
      setNote('');
      setShowNoteInput(false);
    } else {
      setShowNoteInput(true);
    }
  };

  const handleCancel = () => {
    setShowNoteInput(false);
    setNote('');
  };

  if (isAcknowledged) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-green-600 text-lg">✓</span>
          <span className="text-green-800 text-sm font-medium">Alert Acknowledged</span>
        </div>
        <button
          onClick={() => onAcknowledge('')}
          className="text-green-600 hover:text-green-800 text-xs underline"
        >
          Reset
        </button>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
      {!showNoteInput ? (
        <button
          onClick={handleAcknowledge}
          className="w-full bg-primary-500 hover:bg-primary-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
        >
          <span>✓</span>
          <span>Acknowledge Alert</span>
        </button>
      ) : (
        <div className="space-y-3">
          <label className="block">
            <span className="text-gray-700 text-sm font-medium">Clinical Note (Optional)</span>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Document your assessment or intervention..."
              className="mt-1 w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none text-gray-900"
              rows={3}
            />
          </label>
          <div className="flex space-x-2">
            <button
              onClick={handleAcknowledge}
              className="flex-1 bg-primary-500 hover:bg-primary-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
            >
              Confirm
            </button>
            <button
              onClick={handleCancel}
              className="flex-1 bg-gray-300 hover:bg-gray-400 text-gray-800 px-4 py-2 rounded-lg font-medium transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

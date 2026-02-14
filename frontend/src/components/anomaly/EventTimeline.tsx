import { VitalMessage } from '../../types';

interface EventTimelineProps {
  events: VitalMessage[];
  maxEvents?: number;
  onEventClick?: (event: VitalMessage) => void;
}

export const EventTimeline = ({ events, maxEvents = 10, onEventClick }: EventTimelineProps) => {
  const recentEvents = events
    .filter(e => e.event_type !== 'NONE' || e.anomaly_detected)
    .slice(-maxEvents)
    .reverse();

  if (recentEvents.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
        <h3 className="text-gray-900 font-semibold mb-3">Event Timeline</h3>
        <div className="text-center py-4 text-gray-500 text-sm">
          No events detected
        </div>
      </div>
    );
  }

  const getEventColor = (event: VitalMessage) => {
    if (event.anomaly_detected) return 'border-orange-400 bg-orange-50';
    
    switch (event.event_type) {
      case 'HYPOTENSION':
        return 'border-red-400 bg-red-50';
      case 'TACHYCARDIA':
        return 'border-red-400 bg-red-50';
      case 'HYPOXIA':
        return 'border-blue-400 bg-blue-50';
      case 'SEPSIS_ALERT':
        return 'border-purple-400 bg-purple-50';
      default:
        return 'border-gray-300 bg-gray-50';
    }
  };

  const getEventIcon = (event: VitalMessage) => {
    if (event.anomaly_detected) return '⚠️';
    
    switch (event.event_type) {
      case 'HYPOTENSION':
        return '📉';
      case 'TACHYCARDIA':
        return '💓';
      case 'HYPOXIA':
        return '🫁';
      case 'SEPSIS_ALERT':
        return '🦠';
      default:
        return '📌';
    }
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
      <h3 className="text-gray-900 font-semibold mb-3">Event Timeline</h3>
      
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {recentEvents.map((event, index) => (
          <button
            key={`${event.timestamp}-${index}`}
            onClick={() => onEventClick?.(event)}
            className={`w-full text-left p-3 rounded-lg border-l-4 ${getEventColor(event)} transition-all hover:shadow-md hover:scale-[1.01] cursor-pointer`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-2">
                <span className="text-xl">{getEventIcon(event)}</span>
                <div>
                  <div className="flex items-center space-x-2">
                    {event.anomaly_detected && (
                      <span className="text-sm font-bold text-orange-700">
                        {event.anomaly_type?.replace('_', ' ') || 'ANOMALY'}
                      </span>
                    )}
                    {event.event_type !== 'NONE' && (
                      <span className="text-sm font-bold text-gray-800">
                        {event.event_type.replace('_', ' ')}
                      </span>
                    )}
                  </div>
                  
                  <div className="text-xs text-gray-600 mt-1 space-y-1">
                    <div className="flex space-x-3">
                      <span>HR: {event.heart_rate}</span>
                      <span>BP: {event.systolic_bp}/{event.diastolic_bp}</span>
                      <span>SpO2: {event.spo2}%</span>
                    </div>
                    {event.risk_score !== undefined && (
                      <div>Risk: {(event.risk_score * 100).toFixed(0)}%</div>
                    )}
                  </div>
                </div>
              </div>
              
              <span className="text-xs text-gray-500 whitespace-nowrap ml-2">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};
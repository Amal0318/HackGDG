import { VitalMessage, EventType } from '../../types';

interface EventTimelineProps {
  events: VitalMessage[];
  onEventClick?: (event: VitalMessage) => void;
}

const getEventColor = (eventType: EventType): string => {
  switch (eventType) {
    case 'HYPOTENSION':
      return 'bg-red-500';
    case 'TACHYCARDIA':
      return 'bg-orange-500';
    case 'HYPOXIA':
      return 'bg-purple-500';
    case 'SEPSIS_ALERT':
      return 'bg-red-700';
    default:
      return 'bg-gray-400';
  }
};

const getEventIcon = (eventType: EventType): string => {
  switch (eventType) {
    case 'HYPOTENSION':
      return '↓';
    case 'TACHYCARDIA':
      return '⚡';
    case 'HYPOXIA':
      return '💨';
    case 'SEPSIS_ALERT':
      return '⚠️';
    default:
      return '•';
  }
};

export const EventTimeline = ({ events, onEventClick }: EventTimelineProps) => {
  const filteredEvents = events.filter(e => e.event_type !== 'NONE');

  if (filteredEvents.length === 0) {
    return (
      <div className="bg-white p-4 rounded-lg shadow text-center text-gray-500">
        No events recorded
      </div>
    );
  }

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Event Timeline</h3>
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {filteredEvents.map((event, index) => (
          <div
            key={index}
            className="flex items-start space-x-3 cursor-pointer hover:bg-gray-50 p-2 rounded transition-colors"
            onClick={() => onEventClick?.(event)}
          >
            <div className={`flex-shrink-0 w-8 h-8 rounded-full ${getEventColor(event.event_type)} flex items-center justify-center text-white text-sm`}>
              {getEventIcon(event.event_type)}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900">
                {event.event_type}
              </p>
              <p className="text-xs text-gray-500">
                {new Date(event.timestamp).toLocaleString()}
              </p>
              {event.anomaly_detected && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800 mt-1">
                  Anomaly
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

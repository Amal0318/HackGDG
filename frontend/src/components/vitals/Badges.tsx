import { PatientState, EventType } from '../../types';

interface BadgeProps {
  state?: PatientState;
  event?: EventType;
}

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

const getEventBgColor = (event: EventType): string => {
  switch (event) {
    case 'NONE':
      return 'bg-slate-600';
    case 'HYPOTENSION':
      return 'bg-orange-500';
    case 'TACHYCARDIA':
      return 'bg-red-500';
    case 'HYPOXIA':
      return 'bg-blue-500';
    case 'SEPSIS_ALERT':
      return 'bg-purple-500';
    default:
      return 'bg-gray-500';
  }
};

export const StateBadge = ({ state }: { state: PatientState }) => {
  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold text-white ${getStateBgColor(state)}`}>
      {state.replace('_', ' ')}
    </span>
  );
};

export const EventBadge = ({ event }: { event: EventType }) => {
  if (event === 'NONE') return null;
  
  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold text-white ${getEventBgColor(event)}`}>
      {event.replace('_', ' ')}
    </span>
  );
};

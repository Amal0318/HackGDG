import { useState } from 'react';
import { EventType, PatientState } from '../../types';

interface EventFilterProps {
  onFilterChange: (filters: FilterOptions) => void;
}

export interface FilterOptions {
  eventTypes: EventType[];
  states: PatientState[];
  showAnomalies: boolean;
  dateRange: 'all' | 'last-hour' | 'last-day';
}

export const EventFilter = ({ onFilterChange }: EventFilterProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [filters, setFilters] = useState<FilterOptions>({
    eventTypes: ['NONE', 'HYPOTENSION', 'TACHYCARDIA', 'HYPOXIA', 'SEPSIS_ALERT'],
    states: ['STABLE', 'EARLY_DETERIORATION', 'CRITICAL', 'INTERVENTION'],
    showAnomalies: true,
    dateRange: 'all'
  });

  const toggleEventType = (type: EventType) => {
    const newTypes = filters.eventTypes.includes(type)
      ? filters.eventTypes.filter(t => t !== type)
      : [...filters.eventTypes, type];
    
    const newFilters = { ...filters, eventTypes: newTypes };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const toggleState = (state: PatientState) => {
    const newStates = filters.states.includes(state)
      ? filters.states.filter(s => s !== state)
      : [...filters.states, state];
    
    const newFilters = { ...filters, states: newStates };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const toggleAnomalies = () => {
    const newFilters = { ...filters, showAnomalies: !filters.showAnomalies };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const setDateRange = (range: 'all' | 'last-hour' | 'last-day') => {
    const newFilters = { ...filters, dateRange: range };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const resetFilters = () => {
    const defaultFilters: FilterOptions = {
      eventTypes: ['NONE', 'HYPOTENSION', 'TACHYCARDIA', 'HYPOXIA', 'SEPSIS_ALERT'],
      states: ['STABLE', 'EARLY_DETERIORATION', 'CRITICAL', 'INTERVENTION'],
      showAnomalies: true,
      dateRange: 'all'
    };
    setFilters(defaultFilters);
    onFilterChange(defaultFilters);
  };

  const activeFilterCount = 
    (5 - filters.eventTypes.length) +
    (4 - filters.states.length) +
    (filters.showAnomalies ? 0 : 1) +
    (filters.dateRange === 'all' ? 0 : 1);

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <span className="text-gray-900 font-semibold">🔍 Event Filters</span>
          {activeFilterCount > 0 && (
            <span className="bg-primary-500 text-white text-xs px-2 py-1 rounded-full">
              {activeFilterCount} active
            </span>
          )}
        </div>
        <span className="text-gray-500 text-xl">{isExpanded ? '▲' : '▼'}</span>
      </button>

      {/* Filter Options */}
      {isExpanded && (
        <div className="px-4 py-4 space-y-4 border-t border-gray-200">
          {/* Event Types */}
          <div>
            <p className="text-gray-700 text-sm font-medium mb-2">Event Types</p>
            <div className="flex flex-wrap gap-2">
              <FilterChip
                label="None"
                isActive={filters.eventTypes.includes('NONE')}
                onClick={() => toggleEventType('NONE')}
                color="gray"
              />
              <FilterChip
                label="Hypotension"
                isActive={filters.eventTypes.includes('HYPOTENSION')}
                onClick={() => toggleEventType('HYPOTENSION')}
                color="red"
              />
              <FilterChip
                label="Tachycardia"
                isActive={filters.eventTypes.includes('TACHYCARDIA')}
                onClick={() => toggleEventType('TACHYCARDIA')}
                color="yellow"
              />
              <FilterChip
                label="Hypoxia"
                isActive={filters.eventTypes.includes('HYPOXIA')}
                onClick={() => toggleEventType('HYPOXIA')}
                color="blue"
              />
              <FilterChip
                label="Sepsis Alert"
                isActive={filters.eventTypes.includes('SEPSIS_ALERT')}
                onClick={() => toggleEventType('SEPSIS_ALERT')}
                color="red"
              />
            </div>
          </div>

          {/* Patient States */}
          <div>
            <p className="text-gray-700 text-sm font-medium mb-2">Patient States</p>
            <div className="flex flex-wrap gap-2">
              <FilterChip
                label="Stable"
                isActive={filters.states.includes('STABLE')}
                onClick={() => toggleState('STABLE')}
                color="teal"
              />
              <FilterChip
                label="Early Deterioration"
                isActive={filters.states.includes('EARLY_DETERIORATION')}
                onClick={() => toggleState('EARLY_DETERIORATION')}
                color="yellow"
              />
              <FilterChip
                label="Critical"
                isActive={filters.states.includes('CRITICAL')}
                onClick={() => toggleState('CRITICAL')}
                color="red"
              />
              <FilterChip
                label="Intervention"
                isActive={filters.states.includes('INTERVENTION')}
                onClick={() => toggleState('INTERVENTION')}
                color="red"
              />
            </div>
          </div>

          {/* Anomalies Toggle */}
          <div>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={filters.showAnomalies}
                onChange={toggleAnomalies}
                className="w-4 h-4 text-primary-500 rounded focus:ring-primary-500"
              />
              <span className="text-gray-700 text-sm font-medium">Show Anomalies</span>
            </label>
          </div>

          {/* Date Range */}
          <div>
            <p className="text-gray-700 text-sm font-medium mb-2">Time Range</p>
            <div className="flex flex-wrap gap-2">
              <FilterChip
                label="All Time"
                isActive={filters.dateRange === 'all'}
                onClick={() => setDateRange('all')}
                color="blue"
              />
              <FilterChip
                label="Last Hour"
                isActive={filters.dateRange === 'last-hour'}
                onClick={() => setDateRange('last-hour')}
                color="blue"
              />
              <FilterChip
                label="Last 24h"
                isActive={filters.dateRange === 'last-day'}
                onClick={() => setDateRange('last-day')}
                color="blue"
              />
            </div>
          </div>

          {/* Reset Button */}
          {activeFilterCount > 0 && (
            <button
              onClick={resetFilters}
              className="w-full bg-gray-200 hover:bg-gray-300 text-gray-800 px-4 py-2 rounded-lg font-medium transition-colors"
            >
              Reset All Filters
            </button>
          )}
        </div>
      )}
    </div>
  );
};

// Helper component for filter chips
const FilterChip = ({ 
  label, 
  isActive, 
  onClick, 
  color 
}: { 
  label: string; 
  isActive: boolean; 
  onClick: () => void;
  color: 'gray' | 'yellow' | 'red' | 'teal' | 'blue';
}) => {
  const colorClasses = {
    gray: isActive ? 'bg-gray-500 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200',
    yellow: isActive ? 'bg-yellow-500 text-white' : 'bg-yellow-100 text-yellow-700 hover:bg-yellow-200',
    red: isActive ? 'bg-red-500 text-white' : 'bg-red-100 text-red-700 hover:bg-red-200',
    teal: isActive ? 'bg-primary-500 text-white' : 'bg-primary-100 text-primary-700 hover:bg-primary-200',
    blue: isActive ? 'bg-blue-500 text-white' : 'bg-blue-100 text-blue-700 hover:bg-blue-200',
  };

  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${colorClasses[color]}`}
    >
      {label}
    </button>
  );
};

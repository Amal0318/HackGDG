import { Fragment } from 'react';
import { Listbox, Transition } from '@headlessui/react';
import { ChevronDown, Check } from 'lucide-react';
import { getFloorStats } from '../mockData';

interface FloorSelectorProps {
  value: number | 'all';
  onChange: (value: number | 'all') => void;
}

export default function FloorSelector({ value, onChange }: FloorSelectorProps) {
  const options = [
    { value: 'all' as const, label: 'All Floors', beds: 48 },
    { value: 1, label: 'Floor 1', ...getFloorStats(1) },
    { value: 2, label: 'Floor 2', ...getFloorStats(2) },
    { value: 3, label: 'Floor 3', ...getFloorStats(3) }
  ];

  const selected = options.find(opt => opt.value === value) || options[0];

  return (
    <Listbox value={value} onChange={onChange}>
      <div className="relative">
        <Listbox.Button className="relative w-full md:w-48 rounded-lg bg-white py-2 pl-4 pr-10 text-left shadow-md hover:shadow-lg transition-shadow focus-visible-ring cursor-pointer">
          <span className="block truncate font-semibold text-gray-900">
            {selected.label}
          </span>
          <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
            <ChevronDown className="h-4 w-4 text-gray-400" aria-hidden="true" />
          </span>
        </Listbox.Button>
        <Transition
          as={Fragment}
          leave="transition ease-in duration-100"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <Listbox.Options className="absolute z-50 mt-2 max-h-60 w-full overflow-auto rounded-lg bg-white py-1 shadow-xl ring-1 ring-black ring-opacity-5 focus:outline-none">
            {options.map((option) => (
              <Listbox.Option
                key={option.value}
                className={({ active }) =>
                  `relative cursor-pointer select-none py-3 px-4 ${
                    active ? 'bg-primary-light text-primary-dark' : 'text-gray-900'
                  }`
                }
                value={option.value}
              >
                {({ selected: isSelected }) => (
                  <div className="flex items-center justify-between">
                    <div>
                      <span className={`block font-semibold ${isSelected ? 'text-primary-dark' : ''}`}>
                        {option.label}
                      </span>
                      {option.value !== 'all' && 'total' in option && (
                        <span className="text-xs text-gray-500">
                          {option.total} beds • {option.highRisk} high risk • {option.activeAlerts} alerts
                        </span>
                      )}
                    </div>
                    {isSelected && (
                      <Check className="h-5 w-5 text-primary" />
                    )}
                  </div>
                )}
              </Listbox.Option>
            ))}
          </Listbox.Options>
        </Transition>
      </div>
    </Listbox>
  );
}

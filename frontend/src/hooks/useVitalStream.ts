import { useEffect } from 'react';
import { wsService } from '../services/websocket';
import { useStore } from '../store';

export const useVitalStream = () => {
  const addVitalMessage = useStore(state => state.addVitalMessage);
  const setWsConnected = useStore(state => state.setWsConnected);

  useEffect(() => {
    // Connect WebSocket
    wsService.connect();

    // Subscribe to messages
    const unsubscribeMessage = wsService.onMessage((message) => {
      addVitalMessage(message);
    });

    // Subscribe to connection changes
    const unsubscribeConnection = wsService.onConnectionChange((connected) => {
      setWsConnected(connected);
    });

    // Cleanup on unmount
    return () => {
      unsubscribeMessage();
      unsubscribeConnection();
      wsService.disconnect();
    };
  }, [addVitalMessage, setWsConnected]);
};

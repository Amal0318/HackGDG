import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import DashboardLayout from './layouts/DashboardLayout';
import NurseDashboard from './pages/NurseDashboard';
import DoctorDashboard from './pages/DoctorDashboard';
import ChiefDashboard from './pages/ChiefDashboard';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<Navigate to="/nurse" replace />} />
          <Route path="nurse" element={<NurseDashboard />} />
          <Route path="doctor" element={<DoctorDashboard />} />
          <Route path="chief" element={<ChiefDashboard />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

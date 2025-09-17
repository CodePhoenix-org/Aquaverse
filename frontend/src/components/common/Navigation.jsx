import React from 'react';
import { Link } from 'react-router-dom';

const Navigation = () => {
  return (
    <nav className="bg-blue-600 text-white px-6 py-3">
      <div className="flex items-center justify-between">
        <Link to="/" className="text-xl font-bold">
          FloatChat
        </Link>
        <div className="text-sm text-blue-100">
          AI-Powered ARGO Ocean Data Explorer
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
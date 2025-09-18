function History() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      <div className="max-w-6xl mx-auto p-6">
        <h1 className="text-3xl font-bold mb-4">Chat History</h1>
        <div className="bg-white/10 border border-white/20 rounded-xl p-6">
          <p className="text-blue-200 mb-4">Hook this table to your backend history endpoint.</p>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="text-blue-200">
                <tr>
                  <th className="text-left px-3 py-2">#</th>
                  <th className="text-left px-3 py-2">Query</th>
                  <th className="text-left px-3 py-2">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/10">
                {[1,2,3,4,5].map((i) => (
                  <tr key={i}>
                    <td className="px-3 py-2">{i}</td>
                    <td className="px-3 py-2">Example question {i}</td>
                    <td className="px-3 py-2">2024-06-0{i} 10:00</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default History;



const notes = [
  { id: 1, title: 'Groceries', body: 'milk, eggs, coffee' },
  { id: 2, title: 'Ideas', body: 'a notes app that searches as you type' },
];

function search(query) {
  const q = String(query || '').toLowerCase();
  return notes.filter((n) => n.title.toLowerCase().includes(q) || n.body.toLowerCase().includes(q));
}

module.exports = { search };

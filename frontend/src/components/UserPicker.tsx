import { useState } from "react";

/** User selector shared by the Feed and Inspector pages. */
export function UserPicker({
  userId,
  onChange,
  maxUsers,
}: {
  userId: number;
  onChange: (id: number) => void;
  maxUsers: number;
}) {
  const [draft, setDraft] = useState(String(userId));

  const clamp = (n: number) => Math.min(Math.max(1, Math.round(n)), maxUsers);

  return (
    <div className="row">
      <button onClick={() => onChange(clamp(Math.floor(Math.random() * maxUsers) + 1))}>
        Random
      </button>
      <button onClick={() => onChange(clamp(userId + 1))} disabled={userId >= maxUsers}>
        Next
      </button>
      <button onClick={() => onChange(clamp(userId - 1))} disabled={userId <= 1}>
        Prev
      </button>
      <input
        type="text"
        inputMode="numeric"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onChange(clamp(Number(draft) || 1));
        }}
        style={{ width: 92 }}
        aria-label="user id"
      />
      <button onClick={() => onChange(clamp(Number(draft) || 1))}>Go</button>
    </div>
  );
}

import { Link } from "react-router-dom";
import { ChevronRight, FlaskConical } from "lucide-react";
import { ApiError } from "./api";
export function Badge({ value }: { value: string }) {
  return <span className={`badge ${value}`}>{value.replaceAll("_", " ")}</span>;
}
export function ErrorBox({
  error,
  retry,
}: {
  error: Error;
  retry?: () => void;
}) {
  return (
    <div className="notice error" role="alert">
      <strong>{error.message}</strong>
      {error instanceof ApiError && error.requestId && (
        <small>Request {error.requestId}</small>
      )}
      {retry && <button onClick={retry}>Try again</button>}
      {error instanceof ApiError && error.status === 401 && (
        <Link to="/settings">
          Set workspace token <ChevronRight size={14} />
        </Link>
      )}
    </div>
  );
}
export function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="empty">
      <FlaskConical size={28} />
      <p>{children}</p>
    </div>
  );
}
export function Loading() {
  return (
    <div className="loading" role="status">
      Loading research…
    </div>
  );
}
export function Metric({
  label,
  value,
  caption,
}: {
  label: string;
  value: React.ReactNode;
  caption?: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {caption && <small>{caption}</small>}
    </div>
  );
}

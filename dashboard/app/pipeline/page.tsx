import { PipelineStatus } from "@/components/pipeline-status";

export default function PipelinePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-purple-100 mb-2">Pipeline Status</h1>
        <p className="text-purple-200/60">Monitor ETL pipeline progress and metrics.</p>
      </div>

      <PipelineStatus />
    </div>
  );
}


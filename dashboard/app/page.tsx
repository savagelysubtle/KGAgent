import { PipelineStatus } from "@/components/pipeline-status";
import { CrawlerControl } from "@/components/crawler-control";

export default function Home() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-purple-100 mb-2">Overview</h1>
        <p className="text-purple-200/60">System status and active jobs.</p>
      </div>

      {/* System Status - Full Width at Top */}
      <section>
        <h2 className="text-xl font-semibold text-purple-100 mb-4">System Status</h2>
        <PipelineStatus />
      </section>

      {/* Quick Actions */}
      <section>
        <h2 className="text-xl font-semibold text-purple-100 mb-4">Quick Actions</h2>
        <CrawlerControl />
      </section>
    </div>
  );
}

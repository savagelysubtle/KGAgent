import { CrawlerControl } from "@/components/crawler-control";

export default function CrawlerPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-purple-100 mb-2">Crawler Control</h1>
        <p className="text-purple-200/60">Configure and manage web crawler jobs.</p>
      </div>

      <div className="max-w-2xl">
        <CrawlerControl />
      </div>
    </div>
  );
}


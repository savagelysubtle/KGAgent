"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { crawlApi } from "@/lib/api";
import { Loader2 } from "lucide-react";

export function CrawlerControl() {
  const [urls, setUrls] = useState("");
  const [depth, setDepth] = useState(1);
  const [isLoading, setIsLoading] = useState(false);

  const handleCrawl = async () => {
    if (!urls.trim()) {
      toast.error("Please enter at least one URL");
      return;
    }

    const urlList = urls.split("\n").filter((u) => u.trim());
    setIsLoading(true);

    try {
      await crawlApi.startCrawl(urlList, depth);
      toast.success("Crawl job started successfully");
      setUrls("");
    } catch (error) {
      console.error(error);
      toast.error("Failed to start crawl job");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="w-full bg-black/40 border-purple-500/20 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-purple-100">Crawler Control</CardTitle>
        <CardDescription className="text-purple-200/60">
          Initiate web crawling jobs for the Knowledge Graph
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="urls" className="text-purple-100">Target URLs (one per line)</Label>
          <Textarea
            id="urls"
            className="min-h-[120px] bg-black/20 border-purple-500/20 text-purple-100 placeholder:text-purple-200/40"
            placeholder="https://example.com"
            value={urls}
            onChange={(e) => setUrls(e.target.value)}
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="depth" className="text-purple-100">Crawl Depth</Label>
          <Input
            id="depth"
            type="number"
            min={1}
            max={5}
            value={depth}
            onChange={(e) => setDepth(parseInt(e.target.value))}
            className="bg-black/20 border-purple-500/20 text-purple-100"
          />
        </div>
      </CardContent>
      <CardFooter>
        <Button
          onClick={handleCrawl}
          disabled={isLoading}
          className="w-full bg-purple-600 hover:bg-purple-700 text-white shadow-[0_0_20px_rgba(147,51,234,0.3)] transition-all duration-300"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Starting...
            </>
          ) : (
            "Start Crawl Job"
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}


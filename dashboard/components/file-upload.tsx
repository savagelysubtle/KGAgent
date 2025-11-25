"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, File as FileIcon, X, Loader2, CheckCircle2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";
import { uploadApi } from "@/lib/api";

export function FileUpload() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles((prev) => [...prev, ...acceptedFiles]);
  }, []);

  const removeFile = (fileToRemove: File) => {
    setFiles((prev) => prev.filter((file) => file !== fileToRemove));
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const handleUpload = async () => {
    if (files.length === 0) return;

    setUploading(true);
    setProgress(10); // Simulate start

    try {
      await uploadApi.uploadFiles(files);
      setProgress(100);
      toast.success(`Successfully uploaded ${files.length} files for processing.`);
      setFiles([]);
    } catch (error) {
      console.error(error);
      toast.error("Failed to upload files.");
    } finally {
      setUploading(false);
      setTimeout(() => setProgress(0), 1000);
    }
  };

  return (
    <Card className="w-full bg-black/40 border-purple-500/20 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-purple-100">File Upload</CardTitle>
        <CardDescription className="text-purple-200/60">
          Upload documents (PDF, HTML) directly for processing.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors ${
            isDragActive
              ? "border-purple-500 bg-purple-500/10"
              : "border-purple-500/20 hover:border-purple-500/50 hover:bg-purple-500/5"
          }`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="p-4 rounded-full bg-purple-500/10">
              <UploadCloud className="h-10 w-10 text-purple-400" />
            </div>
            <div className="space-y-1">
              <p className="text-lg font-medium text-purple-100">
                {isDragActive ? "Drop files here" : "Drag & drop files here"}
              </p>
              <p className="text-sm text-purple-200/60">
                or click to browse from your computer
              </p>
            </div>
          </div>
        </div>

        {files.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-purple-100">Selected Files</h3>
            <div className="grid gap-2">
              {files.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 rounded-md bg-purple-500/5 border border-purple-500/10 group"
                >
                  <div className="flex items-center space-x-3 overflow-hidden">
                    <FileIcon className="h-5 w-5 text-purple-400 flex-shrink-0" />
                    <span className="text-sm text-purple-100 truncate">
                      {file.name}
                    </span>
                    <span className="text-xs text-purple-200/40 flex-shrink-0">
                      {(file.size / 1024).toFixed(1)} KB
                    </span>
                  </div>
                  <button
                    onClick={() => removeFile(file)}
                    className="p-1 hover:bg-purple-500/20 rounded-full transition-colors"
                  >
                    <X className="h-4 w-4 text-purple-200/60 hover:text-purple-100" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {uploading && (
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-purple-200/60">
              <span>Uploading...</span>
              <span>{progress}%</span>
            </div>
            <Progress value={progress} className="h-2 bg-purple-950/50" />
          </div>
        )}
      </CardContent>
      <CardFooter>
        <Button
          onClick={handleUpload}
          disabled={files.length === 0 || uploading}
          className="w-full bg-purple-600 hover:bg-purple-700 text-white shadow-[0_0_20px_rgba(147,51,234,0.3)] transition-all duration-300"
        >
          {uploading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <UploadCloud className="mr-2 h-4 w-4" />
              Upload & Process
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}


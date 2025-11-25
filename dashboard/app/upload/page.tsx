import { FileUpload } from "@/components/file-upload";

export default function UploadPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-purple-100 mb-2">File Upload</h1>
        <p className="text-purple-200/60">
          Upload local documents to the Knowledge Graph pipeline.
        </p>
      </div>

      <div className="max-w-2xl">
        <FileUpload />
      </div>
    </div>
  );
}


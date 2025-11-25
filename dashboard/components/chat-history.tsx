"use client";

import { useState, useEffect, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { chatApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { toast } from "sonner";
import {
  History,
  Plus,
  MessageSquare,
  Trash2,
  MoreVertical,
  Edit2,
  Search,
  Loader2,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface Conversation {
  id: string;
  title: string;
  summary: string | null;
  message_count: number;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  last_message_at: string | null;
}

interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

interface ChatHistoryProps {
  currentConversationId: string | null;
  onSelectConversation: (conversationId: string | null, messages: Message[]) => void;
  onNewChat: () => void;
}

export function ChatHistory({
  currentConversationId,
  onSelectConversation,
  onNewChat,
}: ChatHistoryProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const queryClient = useQueryClient();

  // Fetch conversations
  const {
    data: conversationsData,
    isLoading: isLoadingConversations,
    refetch: refetchConversations,
  } = useQuery({
    queryKey: ["conversations", search],
    queryFn: async () => {
      const response = await chatApi.listConversations({
        limit: 50,
        search: search || undefined,
      });
      return response.data;
    },
    enabled: isOpen,
  });

  // Create conversation mutation
  const createMutation = useMutation({
    mutationFn: async (title?: string) => {
      return chatApi.createConversation({ title });
    },
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
      onSelectConversation(response.data.conversation.id, []);
      toast.success("New chat created");
    },
    onError: (error) => {
      toast.error(`Failed to create chat: ${error}`);
    },
  });

  // Update conversation mutation
  const updateMutation = useMutation({
    mutationFn: async ({ id, title }: { id: string; title: string }) => {
      return chatApi.updateConversation(id, { title });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
      setEditingId(null);
      toast.success("Chat renamed");
    },
    onError: (error) => {
      toast.error(`Failed to rename: ${error}`);
    },
  });

  // Delete conversation mutation
  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      return chatApi.deleteConversation(id);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
      if (currentConversationId === editingId) {
        onNewChat();
      }
      toast.success("Chat deleted");
    },
    onError: (error) => {
      toast.error(`Failed to delete: ${error}`);
    },
  });

  // Load conversation messages
  const loadConversation = useCallback(
    async (convId: string) => {
      try {
        const response = await chatApi.getMessages(convId, {
          limit: 500,
          order: "asc",
        });
        const messages = response.data.messages || [];
        onSelectConversation(convId, messages);
        setIsOpen(false);
      } catch (error) {
        toast.error("Failed to load conversation");
      }
    },
    [onSelectConversation]
  );

  const handleNewChat = () => {
    onNewChat();
    setIsOpen(false);
  };

  const handleStartEdit = (conv: Conversation) => {
    setEditingId(conv.id);
    setEditTitle(conv.title);
  };

  const handleSaveEdit = () => {
    if (editingId && editTitle.trim()) {
      updateMutation.mutate({ id: editingId, title: editTitle.trim() });
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditTitle("");
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return "Today";
    } else if (diffDays === 1) {
      return "Yesterday";
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const conversations: Conversation[] = conversationsData?.conversations || [];

  // Group conversations by date
  const groupedConversations = conversations.reduce(
    (groups, conv) => {
      const dateLabel = formatDate(conv.updated_at);
      if (!groups[dateLabel]) {
        groups[dateLabel] = [];
      }
      groups[dateLabel].push(conv);
      return groups;
    },
    {} as Record<string, Conversation[]>
  );

  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="fixed bottom-24 right-6 z-40 bg-black/80 border border-purple-500/30 hover:bg-purple-500/20 shadow-lg"
          title="Chat History"
        >
          <History className="h-5 w-5 text-purple-400" />
        </Button>
      </SheetTrigger>
      <SheetContent
        side="left"
        className="w-80 bg-black/95 border-purple-500/30 p-0"
      >
        <SheetHeader className="p-4 border-b border-purple-500/20">
          <SheetTitle className="text-purple-300 flex items-center gap-2">
            <History className="h-5 w-5" />
            Chat History
          </SheetTitle>
        </SheetHeader>

        <div className="p-3 border-b border-purple-500/20">
          <Button
            onClick={handleNewChat}
            className="w-full bg-purple-600 hover:bg-purple-700 text-white"
          >
            <Plus className="h-4 w-4 mr-2" />
            New Chat
          </Button>
        </div>

        <div className="p-3 border-b border-purple-500/20">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search chats..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-10 bg-black/40 border-purple-500/30 text-sm"
            />
          </div>
        </div>

        <ScrollArea className="h-[calc(100vh-200px)]">
          {isLoadingConversations ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-purple-400" />
            </div>
          ) : conversations.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <MessageSquare className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p>No conversations yet</p>
              <p className="text-sm">Start a new chat to begin</p>
            </div>
          ) : (
            <div className="p-2">
              {Object.entries(groupedConversations).map(([dateLabel, convs]) => (
                <div key={dateLabel} className="mb-4">
                  <div className="text-xs text-gray-500 px-2 py-1 uppercase tracking-wider">
                    {dateLabel}
                  </div>
                  {convs.map((conv) => (
                    <div
                      key={conv.id}
                      className={cn(
                        "group flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-colors",
                        currentConversationId === conv.id
                          ? "bg-purple-500/20 border border-purple-500/40"
                          : "hover:bg-purple-500/10"
                      )}
                    >
                      {editingId === conv.id ? (
                        <div className="flex-1 flex items-center gap-1">
                          <Input
                            value={editTitle}
                            onChange={(e) => setEditTitle(e.target.value)}
                            className="h-7 text-sm bg-black/40 border-purple-500/30"
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === "Enter") handleSaveEdit();
                              if (e.key === "Escape") handleCancelEdit();
                            }}
                          />
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-7 w-7"
                            onClick={handleSaveEdit}
                          >
                            <Edit2 className="h-3 w-3 text-green-400" />
                          </Button>
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-7 w-7"
                            onClick={handleCancelEdit}
                          >
                            <X className="h-3 w-3 text-red-400" />
                          </Button>
                        </div>
                      ) : (
                        <>
                          <div
                            className="flex-1 min-w-0"
                            onClick={() => loadConversation(conv.id)}
                          >
                            <div className="flex items-center gap-2">
                              <MessageSquare className="h-4 w-4 text-purple-400 shrink-0" />
                              <span className="text-sm text-gray-200 truncate">
                                {conv.title}
                              </span>
                            </div>
                            <div className="text-xs text-gray-500 ml-6">
                              {conv.message_count} messages
                            </div>
                          </div>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button
                                size="icon"
                                variant="ghost"
                                className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
                              >
                                <MoreVertical className="h-4 w-4 text-gray-400" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent
                              align="end"
                              className="bg-black/95 border-purple-500/30"
                            >
                              <DropdownMenuItem
                                onClick={() => handleStartEdit(conv)}
                                className="text-gray-200 focus:bg-purple-500/20"
                              >
                                <Edit2 className="h-4 w-4 mr-2" />
                                Rename
                              </DropdownMenuItem>
                              <AlertDialog>
                                <AlertDialogTrigger asChild>
                                  <DropdownMenuItem
                                    onSelect={(e) => e.preventDefault()}
                                    className="text-red-400 focus:bg-red-500/20"
                                  >
                                    <Trash2 className="h-4 w-4 mr-2" />
                                    Delete
                                  </DropdownMenuItem>
                                </AlertDialogTrigger>
                                <AlertDialogContent className="bg-black/95 border-purple-500/30">
                                  <AlertDialogHeader>
                                    <AlertDialogTitle className="text-gray-200">
                                      Delete Chat?
                                    </AlertDialogTitle>
                                    <AlertDialogDescription>
                                      This will permanently delete &quot;{conv.title}&quot;
                                      and all its messages. This action cannot be
                                      undone.
                                    </AlertDialogDescription>
                                  </AlertDialogHeader>
                                  <AlertDialogFooter>
                                    <AlertDialogCancel className="border-purple-500/30">
                                      Cancel
                                    </AlertDialogCancel>
                                    <AlertDialogAction
                                      onClick={() => deleteMutation.mutate(conv.id)}
                                      className="bg-red-600 hover:bg-red-700"
                                    >
                                      Delete
                                    </AlertDialogAction>
                                  </AlertDialogFooter>
                                </AlertDialogContent>
                              </AlertDialog>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

// Hook to manage chat history state
export function useChatHistory() {
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const queryClient = useQueryClient();

  // Save message to current conversation
  const saveMessage = useCallback(
    async (role: "user" | "assistant", content: string) => {
      if (!currentConversationId) {
        // Create new conversation if none exists
        try {
          const response = await chatApi.createConversation({
            title: content.slice(0, 50) + (content.length > 50 ? "..." : ""),
          });
          const newConvId = response.data.conversation.id;
          setCurrentConversationId(newConvId);

          // Add message to new conversation
          await chatApi.addMessage(newConvId, { role, content });
          queryClient.invalidateQueries({ queryKey: ["conversations"] });
          return newConvId;
        } catch (error) {
          console.error("Failed to create conversation:", error);
          return null;
        }
      } else {
        // Add to existing conversation
        try {
          await chatApi.addMessage(currentConversationId, { role, content });
          queryClient.invalidateQueries({ queryKey: ["conversations"] });
          return currentConversationId;
        } catch (error) {
          console.error("Failed to save message:", error);
          return currentConversationId;
        }
      }
    },
    [currentConversationId, queryClient]
  );

  const selectConversation = useCallback(
    (convId: string | null, loadedMessages: Message[]) => {
      setCurrentConversationId(convId);
      setMessages(loadedMessages);
    },
    []
  );

  const startNewChat = useCallback(() => {
    setCurrentConversationId(null);
    setMessages([]);
  }, []);

  return {
    currentConversationId,
    messages,
    saveMessage,
    selectConversation,
    startNewChat,
    setMessages,
  };
}


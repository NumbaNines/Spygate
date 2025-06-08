import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { analysisService, GameAnalysis } from "@/api/services/analysis";

export function useAnalyses(params?: Record<string, any>) {
  return useQuery({
    queryKey: ["analyses", params],
    queryFn: () => analysisService.getAnalyses(params),
  });
}

export function useAnalysis(id: number) {
  return useQuery({
    queryKey: ["analysis", id],
    queryFn: () => analysisService.getAnalysis(id),
    enabled: !!id,
  });
}

export function useCreateAnalysis() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: FormData) => analysisService.createAnalysis(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["analyses"] });
    },
  });
}

export function useUpdateAnalysis() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: Partial<GameAnalysis> }) =>
      analysisService.updateAnalysis(id, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["analyses"] });
      queryClient.invalidateQueries({ queryKey: ["analysis", variables.id] });
    },
  });
}

export function useDeleteAnalysis() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => analysisService.deleteAnalysis(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["analyses"] });
    },
  });
}

export function useReprocessAnalysis() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => analysisService.reprocessAnalysis(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ["analysis", id] });
    },
  });
}

export function useFormations(analysisId: number) {
  return useQuery({
    queryKey: ["formations", analysisId],
    queryFn: () => analysisService.getFormations(analysisId),
    enabled: !!analysisId,
  });
}

export function usePlays(analysisId: number) {
  return useQuery({
    queryKey: ["plays", analysisId],
    queryFn: () => analysisService.getPlays(analysisId),
    enabled: !!analysisId,
  });
}

export function useSituations(analysisId: number) {
  return useQuery({
    queryKey: ["situations", analysisId],
    queryFn: () => analysisService.getSituations(analysisId),
    enabled: !!analysisId,
  });
}

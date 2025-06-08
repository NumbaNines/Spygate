import React from "react";
import { MainLayout } from "@/components/layout/MainLayout";
import {
  useAnalysis,
  useFormations,
  usePlays,
  useSituations,
} from "@/hooks/useAnalysis";
import { useFeature } from "@/hooks/useMetrics";
import {
  PlayIcon,
  PauseIcon,
  ArrowPathIcon,
  ChartBarIcon,
  DocumentTextIcon,
} from "@heroicons/react/24/outline";

interface AnalysisPageProps {
  params: {
    id: string;
  };
}

export default function AnalysisPage({ params }: AnalysisPageProps) {
  const analysisId = parseInt(params.id);
  const { data: analysis, isLoading: analysisLoading } =
    useAnalysis(analysisId);
  const { data: formations } = useFormations(analysisId);
  const { data: plays } = usePlays(analysisId);
  const { data: situations } = useSituations(analysisId);
  const { trackUsage } = useFeature("view_analysis");

  React.useEffect(() => {
    if (analysis) {
      trackUsage();
    }
  }, [analysis, trackUsage]);

  if (analysisLoading) {
    return (
      <MainLayout>
        <div className="flex h-96 items-center justify-center">
          <div className="text-center">
            <ArrowPathIcon className="mx-auto h-12 w-12 animate-spin text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">
              Loading...
            </h3>
          </div>
        </div>
      </MainLayout>
    );
  }

  if (!analysis) {
    return (
      <MainLayout>
        <div className="flex h-96 items-center justify-center">
          <div className="text-center">
            <h3 className="text-lg font-medium text-gray-900">
              Analysis not found
            </h3>
          </div>
        </div>
      </MainLayout>
    );
  }

  return (
    <MainLayout>
      <div className="py-6">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 md:px-8">
          <h1 className="text-2xl font-semibold text-gray-900">
            {analysis.title}
          </h1>
          <p className="mt-1 text-sm text-gray-500">{analysis.description}</p>
        </div>

        <div className="mx-auto mt-8 max-w-7xl px-4 sm:px-6 md:px-8">
          {/* Video Player */}
          <div className="overflow-hidden rounded-lg bg-black shadow">
            <div className="aspect-w-16 aspect-h-9">
              <video
                src={analysis.video_file}
                controls
                className="h-full w-full object-cover"
              />
            </div>
          </div>

          {/* Analysis Stats */}
          <div className="mt-8 grid grid-cols-1 gap-5 sm:grid-cols-3">
            <div className="overflow-hidden rounded-lg bg-white shadow">
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <ChartBarIcon className="h-6 w-6 text-gray-400" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="truncate text-sm font-medium text-gray-500">
                        Formations Detected
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        {formations?.length || 0}
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
            </div>

            <div className="overflow-hidden rounded-lg bg-white shadow">
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <PlayIcon className="h-6 w-6 text-gray-400" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="truncate text-sm font-medium text-gray-500">
                        Plays Analyzed
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        {plays?.length || 0}
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
            </div>

            <div className="overflow-hidden rounded-lg bg-white shadow">
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <DocumentTextIcon className="h-6 w-6 text-gray-400" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="truncate text-sm font-medium text-gray-500">
                        Situations Identified
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        {situations?.length || 0}
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Formations List */}
          <div className="mt-8">
            <div className="sm:flex sm:items-center">
              <div className="sm:flex-auto">
                <h2 className="text-xl font-semibold text-gray-900">
                  Formations
                </h2>
                <p className="mt-2 text-sm text-gray-700">
                  A list of all formations detected in the game.
                </p>
              </div>
            </div>
            <div className="mt-8 flex flex-col">
              <div className="-my-2 -mx-4 overflow-x-auto sm:-mx-6 lg:-mx-8">
                <div className="inline-block min-w-full py-2 align-middle md:px-6 lg:px-8">
                  <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                    <table className="min-w-full divide-y divide-gray-300">
                      <thead className="bg-gray-50">
                        <tr>
                          <th
                            scope="col"
                            className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6"
                          >
                            Name
                          </th>
                          <th
                            scope="col"
                            className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900"
                          >
                            Type
                          </th>
                          <th
                            scope="col"
                            className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900"
                          >
                            Confidence
                          </th>
                          <th
                            scope="col"
                            className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900"
                          >
                            Timestamp
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200 bg-white">
                        {formations?.map((formation) => (
                          <tr key={formation.id}>
                            <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">
                              {formation.name}
                            </td>
                            <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                              {formation.formation_type}
                            </td>
                            <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                              {(formation.confidence_score * 100).toFixed(1)}%
                            </td>
                            <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                              {new Date(formation.timestamp * 1000)
                                .toISOString()
                                .substr(11, 8)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Plays List */}
          <div className="mt-8">
            <div className="sm:flex sm:items-center">
              <div className="sm:flex-auto">
                <h2 className="text-xl font-semibold text-gray-900">Plays</h2>
                <p className="mt-2 text-sm text-gray-700">
                  A list of all plays detected in the game.
                </p>
              </div>
            </div>
            <div className="mt-8 flex flex-col">
              <div className="-my-2 -mx-4 overflow-x-auto sm:-mx-6 lg:-mx-8">
                <div className="inline-block min-w-full py-2 align-middle md:px-6 lg:px-8">
                  <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                    <table className="min-w-full divide-y divide-gray-300">
                      <thead className="bg-gray-50">
                        <tr>
                          <th
                            scope="col"
                            className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6"
                          >
                            Name
                          </th>
                          <th
                            scope="col"
                            className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900"
                          >
                            Type
                          </th>
                          <th
                            scope="col"
                            className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900"
                          >
                            Success Rate
                          </th>
                          <th
                            scope="col"
                            className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900"
                          >
                            Yards Gained
                          </th>
                          <th
                            scope="col"
                            className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900"
                          >
                            Duration
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200 bg-white">
                        {plays?.map((play) => (
                          <tr key={play.id}>
                            <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">
                              {play.name}
                            </td>
                            <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                              {play.play_type}
                            </td>
                            <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                              {play.success_rate
                                ? `${(play.success_rate * 100).toFixed(1)}%`
                                : "N/A"}
                            </td>
                            <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                              {play.yards_gained ?? "N/A"}
                            </td>
                            <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                              {(
                                (play.end_time - play.start_time) /
                                1000
                              ).toFixed(1)}
                              s
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </MainLayout>
  );
}

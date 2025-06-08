import React from "react";
import { MainLayout } from "@/components/layout/MainLayout";
import { useAnalyses } from "@/hooks/useAnalysis";
import { useUserMetrics } from "@/hooks/useMetrics";
import { GameAnalysis } from "@/api/services/analysis";
import {
  ChartBarIcon,
  VideoCameraIcon,
  ClockIcon,
  ArrowUpIcon,
} from "@heroicons/react/24/outline";

export default function DashboardPage() {
  const { data: analyses, isLoading: analysesLoading } = useAnalyses();
  const { data: metrics, isLoading: metricsLoading } = useUserMetrics();

  const stats = [
    {
      name: "Total Analyses",
      value: metrics?.total_analyses || 0,
      icon: ChartBarIcon,
      change: "+4.75%",
      changeType: "positive",
    },
    {
      name: "Total Videos",
      value: metrics?.total_videos || 0,
      icon: VideoCameraIcon,
      change: "+54.02%",
      changeType: "positive",
    },
    {
      name: "Storage Used",
      value: `${(metrics?.total_storage_used || 0) / 1024 / 1024} GB`,
      icon: ClockIcon,
      change: "+2.45%",
      changeType: "positive",
    },
  ];

  const recentAnalyses =
    (analyses?.results as GameAnalysis[] | undefined)?.slice(0, 5) || [];

  return (
    <MainLayout>
      <div className="py-6">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 md:px-8">
          <h1 className="text-2xl font-semibold text-gray-900">Dashboard</h1>
        </div>
        <div className="mx-auto max-w-7xl px-4 sm:px-6 md:px-8">
          {/* Stats */}
          <div className="mt-8">
            <dl className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-3">
              {stats.map((item) => (
                <div
                  key={item.name}
                  className="relative overflow-hidden rounded-lg bg-white px-4 pt-5 pb-12 shadow sm:px-6 sm:pt-6"
                >
                  <dt>
                    <div className="absolute rounded-md bg-blue-500 p-3">
                      <item.icon
                        className="h-6 w-6 text-white"
                        aria-hidden="true"
                      />
                    </div>
                    <p className="ml-16 truncate text-sm font-medium text-gray-500">
                      {item.name}
                    </p>
                  </dt>
                  <dd className="ml-16 flex items-baseline pb-6 sm:pb-7">
                    <p className="text-2xl font-semibold text-gray-900">
                      {item.value}
                    </p>
                    <p
                      className={`ml-2 flex items-baseline text-sm font-semibold ${
                        item.changeType === "positive"
                          ? "text-green-600"
                          : "text-red-600"
                      }`}
                    >
                      <ArrowUpIcon
                        className="h-5 w-5 flex-shrink-0 self-center text-green-500"
                        aria-hidden="true"
                      />
                      <span className="sr-only">
                        {item.changeType === "positive"
                          ? "Increased"
                          : "Decreased"}{" "}
                        by
                      </span>
                      {item.change}
                    </p>
                  </dd>
                </div>
              ))}
            </dl>
          </div>

          {/* Recent Analyses */}
          <div className="mt-8">
            <div className="sm:flex sm:items-center">
              <div className="sm:flex-auto">
                <h2 className="text-xl font-semibold text-gray-900">
                  Recent Analyses
                </h2>
                <p className="mt-2 text-sm text-gray-700">
                  A list of your most recent game analyses.
                </p>
              </div>
              <div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none">
                <button
                  type="button"
                  className="inline-flex items-center justify-center rounded-md border border-transparent bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 sm:w-auto"
                >
                  New Analysis
                </button>
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
                            Title
                          </th>
                          <th
                            scope="col"
                            className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900"
                          >
                            Status
                          </th>
                          <th
                            scope="col"
                            className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900"
                          >
                            Created
                          </th>
                          <th
                            scope="col"
                            className="relative py-3.5 pl-3 pr-4 sm:pr-6"
                          >
                            <span className="sr-only">Actions</span>
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200 bg-white">
                        {recentAnalyses.map((analysis: GameAnalysis) => (
                          <tr key={analysis.id}>
                            <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">
                              {analysis.title}
                            </td>
                            <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                              <span
                                className={`inline-flex rounded-full px-2 text-xs font-semibold leading-5 ${
                                  analysis.processing_status === "completed"
                                    ? "bg-green-100 text-green-800"
                                    : analysis.processing_status ===
                                      "processing"
                                    ? "bg-yellow-100 text-yellow-800"
                                    : analysis.processing_status === "failed"
                                    ? "bg-red-100 text-red-800"
                                    : "bg-gray-100 text-gray-800"
                                }`}
                              >
                                {analysis.processing_status}
                              </span>
                            </td>
                            <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                              {new Date(
                                analysis.created_at
                              ).toLocaleDateString()}
                            </td>
                            <td className="relative whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm font-medium sm:pr-6">
                              <a
                                href={`/analysis/${analysis.id}`}
                                className="text-blue-600 hover:text-blue-900"
                              >
                                View
                              </a>
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

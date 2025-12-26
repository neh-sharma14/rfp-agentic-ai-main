/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import moment from "moment-timezone";
import { CandlestickChart, LineChart, BarChart } from "echarts/charts";
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  DataZoomComponent,
  LegendComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import { DataPoint } from "src/helpers";
import {
  formatNumber,
  formatNumberWithSuffix,
  getMonthName,
} from "pages/api/helper_func/sharedFunction";

echarts.use([
  CandlestickChart,
  LineChart,
  BarChart,
  TitleComponent,
  TooltipComponent,
  GridComponent,
  DataZoomComponent,
  LegendComponent,
  CanvasRenderer,
]);

type BarData = { value: number };
type CandlestickData = [number, number, number, number, number]; // [timestamp, open, close, low, high]
type LineData = number;

const Candle = ({
  candleData,
  darkMode,
  isExpand,
  isAnalyzeStock,
  obDate,
  retestTouchDate,
  isOrderblock,
  fromDate,
  toDate,
  transactionType,
  isOrderBlock,
}: {
  candleData: DataPoint[];
  darkMode: boolean;
  isExpand?: boolean;
  isOrderblock?: boolean;
  isAnalyzeStock?: boolean;
  obDate?: string | undefined;
  fromDate?: string | undefined;
  retestTouchDate?: string | undefined;
  toDate?: string | undefined;
  transactionType?: string;
  isOrderBlock?: boolean;
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.EChartsType | null>(null);

  const getVolumes = (data: typeof candleData) => {
    return data.map((item) => ({
      value: +item.volume,
      itemStyle: {
        color: item.close > item.open ? "#089981" : "#FD1050",
      },
    }));
  };

  const dates = candleData.map((item) => {
    const d = new Date(item.date);
    return `${d.getFullYear()}-${(d.getMonth() + 1)
      .toString()
      .padStart(2, "0")}-${d.getDate().toString().padStart(2, "0")}`;
  });

  const normalizeDate = (dateStr?: string) => {
    if (!dateStr) return undefined;
    const d = new Date(dateStr);
    return `${d.getFullYear()}-${(d.getMonth() + 1)
      .toString()
      .padStart(2, "0")}-${d.getDate().toString().padStart(2, "0")}`;
  };

  const formattedOBDate = normalizeDate(obDate);
  const formattedRetestDate = normalizeDate(retestTouchDate);

  const obIndex = formattedOBDate
    ? dates.findIndex((d) => d === formattedOBDate)
    : -1;

  const retestIndex = formattedRetestDate
    ? dates.findIndex((d) => d === formattedRetestDate)
    : -1;

  const ohlc = candleData.map((item) => [
    item.open,
    item.close,
    item.low,
    item.high,
  ]);

  const lastDate = new Date(dates[dates.length - 1]);
  const threeMonthsAgo = new Date(lastDate);
  threeMonthsAgo.setMonth(lastDate.getMonth() - 3);

  const getLabelText = (type: string | null | undefined) => {
    if (!type) return "";
    const lower = type.toLowerCase();

    if (lower.includes("sell") || lower.includes("sold")) {
      return "Sold here";
    }
    if (lower.includes("buy") || lower.includes("purchase")) {
      return "Bought here";
    }

    return type;
  };

  let startIndex = dates.findIndex((d) => new Date(d) >= threeMonthsAgo);
  if (startIndex === -1) startIndex = 0;

  const end = 100;
  const start = (startIndex / dates.length) * 100;
  const symbolName = "Candle";

  const orderBlockMarkArea: any[] = [];

  if (obIndex !== -1 && ohlc[obIndex]) {
    const obCandle = ohlc[obIndex];
    const obLow = obCandle[2];
    const obHigh = obCandle[3];
    const obEndIndex =
      retestIndex !== -1 && retestIndex > obIndex
        ? retestIndex
        : dates.length - 1;
    orderBlockMarkArea.push([
      {
        name: "Order Block Zone",
        itemStyle: {
          color: darkMode
            ? "rgba(0, 180, 255, 0.3)"
            : "rgba(50, 150, 255, 0.2)",
        },
        label: { show: false },
        xAxis: dates[obIndex],
        yAxis: obLow,
      },
      {
        xAxis: dates[obEndIndex],
        yAxis: obHigh,
      },
    ]);
    orderBlockMarkArea.push([
      {
        name: "OB Created",
        itemStyle: {
          color: "transparent",
          borderColor: darkMode ? "#00B4FF" : "#0966C8",
          borderWidth: 1,
        },
        label: { show: false },
        xAxis: dates[obIndex],
        yAxis: obLow,
      },
      {
        xAxis: dates[obIndex],
        yAxis: obHigh,
      },
    ]);
  }

  if (retestIndex !== -1 && ohlc[retestIndex]) {
    const retestCandle = ohlc[retestIndex];
    const retestLow = retestCandle[2];
    const retestHigh = retestCandle[3];
    orderBlockMarkArea.push([
      {
        name: "Retest Touch",
        itemStyle: {
          color: "transparent",
          borderColor: darkMode ? "#FFA500" : "#FF8C00",
          borderWidth: 1,
        },
        label: { show: false },
        xAxis: dates[retestIndex],
        yAxis: retestLow,
      },
      {
        xAxis: dates[retestIndex],
        yAxis: retestHigh,
      },
    ]);
  }

  const normalizedFrom = normalizeDate(fromDate);
  const normalizedTo = normalizeDate(toDate);
  const candlestickMarkPoints: any[] = [];

  if (normalizedFrom) {
    const fromIndex = dates.findIndex((d) => d === normalizedFrom);
    const toIndex = normalizedTo
      ? dates.findIndex((d) => d === normalizedTo)
      : -1;
    if (fromIndex !== -1) {
      candlestickMarkPoints.push({
        name: transactionType || "Start",
        coord: [dates[fromIndex], ohlc[fromIndex][3]],
        symbol: "pin",
        symbolSize: 16,
        itemStyle: {
          color: darkMode ? "#8f8d7bff" : "#ada997ff",
          shadowBlur: 10,
          shadowColor: darkMode ? "#fff" : "#333",
        },
        label: {
          show: true,
          position: "top",
          formatter: getLabelText(transactionType),
          color: "#fff",
          backgroundColor: darkMode ? "rgba(255,255,255,0.2)" : "#000",
          borderRadius: 4,
          padding: [4, 6],
          fontSize: 12,
        },
      });

      if (toIndex !== -1 && toIndex !== fromIndex) {
        candlestickMarkPoints.push({
          name: transactionType || "End",
          coord: [dates[toIndex], ohlc[toIndex][3]],
          symbol: "pin",
          symbolSize: 16,
          itemStyle: {
            color: darkMode ? "#8f8d7bff" : "#ada997ff",
            shadowBlur: 10,
            shadowColor: darkMode ? "#fff" : "#333",
          },
          label: {
            show: true,
            position: fromIndex === toIndex - 1 ? "bottom" : "top", // <-- adjust if adjacent
            formatter: getLabelText(transactionType),
            color: "#fff",
            backgroundColor: darkMode ? "rgba(255,255,255,0.2)" : "#000",
            borderRadius: 4,
            padding: [4, 6],
            fontSize: 12,
          },
        });
      }
    }
  }

  const maColors = {
    MA20: undefined, // default color
    MA50: !darkMode ? "#19C37D" : undefined,
    MA100: !darkMode ? "#FA7B17" : undefined,
    MA200: !darkMode ? "#FD1050" : undefined,
  };

  const legendData = [
    { name: "MA20", itemStyle: { color: maColors.MA20 } },
    { name: "MA50", itemStyle: { color: maColors.MA50 } },
    { name: "MA100", itemStyle: { color: maColors.MA100 } },
    { name: "MA200", itemStyle: { color: maColors.MA200 } },
    symbolName,
    "Volume",
  ];

  const option = {
    backgroundColor: darkMode ? "#121419" : "#ffffff",
    legend: {
      data: legendData,
      selected: {
        MA20: false,
        MA50: false,
        MA100: false,
        MA200: false,
        Volume: false,
      },
      inactiveColor: "#777",
      itemWidth: 14,
      itemHeight: 7,
    },
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "cross",
      },
      confine: true,
      formatter: (
        params: {
          axisValue: string;
          seriesType: string;
          seriesName: string;
          color: string;
          data: BarData | CandlestickData | LineData;
          dataIndex: number;
        }[],
      ) => {
        const dateStr = params[0]?.axisValue;
        const dateIST = moment
          .tz(dateStr, "UTC")
          .tz("Asia/Kolkata")
          .format("DD MMM YYYY");

        let tooltipHtml = `<div>${dateIST}</div>`;
        const fmt = (val: number) =>
          Number(val).toLocaleString("en-US", {
            minimumFractionDigits: 0,
            maximumFractionDigits: 2,
          });

        params.forEach((param) => {
          if (param.seriesType === "candlestick") {
            const idx = param.dataIndex as number;
            const candle: any = candleData[idx];

            tooltipHtml += `
      <div>
        Open: ${fmt(candle.open)}<br/>
        High: ${fmt(candle.high)}<br/>
        Low: ${fmt(candle.low)}<br/>
        Close: ${fmt(candle.close)}<br/>
        Volume: ${formatNumberWithSuffix(candle.volume)}
      </div>`;
          } else if (param.seriesType === "line") {
            const lineData = param.data as LineData;
            const formattedLineValue = fmt(Number(lineData));

            tooltipHtml += `
      <div>
        <span style="color:${param.color}; margin-right:6px;">●</span>
        <b>${param.seriesName}: ${formattedLineValue}</b>
      </div>`;
          }
        });

        return tooltipHtml;
      },
    },

    axisPointer: {
      link: { xAxisIndex: "all" },
    },
    grid: [
      {
        left:
          isAnalyzeStock && !isExpand
            ? 30
            : isOrderblock && !isExpand
              ? 12
              : 30,
        top: "16%",
        height: isOrderblock && !isExpand ? "68%" : "70%",
        width: isExpand ? "94%" : "90%",
      },
      {
        left:
          isAnalyzeStock && !isExpand
            ? 30
            : isOrderblock && !isExpand
              ? 12
              : 30,
        bottom: isExpand ? 56 : isAnalyzeStock ? 35 : isOrderblock ? 32 : 47,
        height: "13%",
        width: isExpand ? "94%" : "90%",
        containLabel: false,
      },
    ],
    xAxis: [
      {
        type: "category",
        gridIndex: 0,
        data: dates,
        axisLine: { show: true },
        axisTick: { show: true },
        splitLine: { show: false },
        axisLabel: {
          rotate: 35,
          formatter: (value: string) => {
            const date = new Date(value);
            return getMonthName(date);
          },
          interval: (index: number, value: string) => {
            const current = new Date(value);
            if (index === 0) return true;
            const prev = new Date(dates[index - 1]);
            return current.getMonth() !== prev.getMonth();
          },
        },
      },
      {
        type: "category",
        gridIndex: 1,
        data: dates,
        axisLine: { show: false },
        axisPointer: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
      },
    ],
    yAxis: [
      {
        scale: true,
        splitLine: { show: false },
        position: "right",
        gridIndex: 0,
        axisLabel: {
          formatter: (value: number) => formatNumber(value, true),
        },
      },
      {
        gridIndex: 1,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { show: false },
        splitLine: { show: false },
      },
    ],
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: [0, 1],
        ...((isExpand || isOrderBlock) && { start, end }),
        minSpan: 30,
        zoomLock: !isExpand,
      },
      {
        type: "slider",
        xAxisIndex: [0, 1],
        ...((isExpand || isOrderBlock) && { start, end }),
        show: false,
        zoomLock: !isExpand,
        minSpan: 30,
      },
    ],
    series: [
      {
        type: "candlestick",
        name: symbolName,
        data: ohlc,
        xAxisIndex: 0,
        yAxisIndex: 0,
        itemStyle: {
          color: "#089981",
          color0: "#FD1050",
          borderColor: "#089981",
          borderColor0: "#FD1050",
        },
        markArea: {
          data: [...orderBlockMarkArea],
        },
        markPoint: {
          symbolKeepAspect: true,
          data: candlestickMarkPoints,
        },
      },
      {
        name: "MA20",
        type: "line",
        data: candleData.map((item) => item["20daysmaline"]),
        smooth: true,
        showSymbol: false,
        lineStyle: {
          width: 1,
        },
      },
      {
        name: "MA50",
        type: "line",
        data: candleData.map((item) => item["50daysmaline"]),
        smooth: true,
        showSymbol: false,
        lineStyle: {
          width: 1,
          color: !darkMode ? "#19C37D" : undefined,
        },
      },
      {
        name: "MA100",
        type: "line",
        data: candleData.map((item) => item["100daysmaline"]),
        smooth: true,
        showSymbol: false,
        lineStyle: {
          width: 1,
          color: !darkMode ? "#FA7B17" : undefined,
        },
      },
      {
        name: "MA200",
        type: "line",
        data: candleData.map((item) => item["200daysmaline"]),
        smooth: true,
        showSymbol: false,
        lineStyle: {
          width: 1,
          color: !darkMode ? "#FD1050" : undefined,
        },
      },

      {
        name: "Volume",
        type: "bar",
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: getVolumes(candleData),
        itemStyle: {
          opacity: 0.3,
        },
      },
    ],
  };

  useEffect(() => {
    if (chartRef.current) {
      if (!chartInstance.current) {
        chartInstance.current = echarts.init(
          chartRef.current,
          darkMode ? "dark" : "light",
        );
      }
      chartInstance.current.setOption(option, true); // <— use 'true' to fully refresh
    }

    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chartInstance.current?.dispose();
      chartInstance.current = null;
    };
  }, [candleData, darkMode, obDate, retestTouchDate, fromDate, toDate]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div
      ref={chartRef}
      id="candle-chart"
      className={`w-full ${isAnalyzeStock || isExpand ? "h-full" : isOrderblock ? "h-[26vh]" : "h-85"}`}
      style={{ transform: "scale(0.98)" }}
    />
  );
};

export default React.memo(Candle);

const API_URL = "https://walmart-sales-forecasting-mlops2.onrender.com/predict";

const upload = document.getElementById("csvUpload")

let dataset = []

upload.addEventListener("change", async function(){

const file = upload.files[0]

const text = await file.text()

dataset = parseCSV(text)

sendForPrediction(dataset)

})

function parseCSV(text){

const rows = text.split("\n")
const headers = rows[0].split(",")

return rows.slice(1).map(row=>{
const values=row.split(",")

let obj={}
headers.forEach((h,i)=>obj[h]=values[i])

return obj
})

}

async function sendForPrediction(data){

const res = await fetch("https://your-api-url/predict",{

method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify(data)

})

const result = await res.json()

renderCharts(result)

}

function renderCharts(data){

let sales = data.map(x=>x.prediction)

let dates = data.map(x=>x.Date)

let total = sales.reduce((a,b)=>a+b,0)

let avg = total/sales.length

let peak = Math.max(...sales)

document.getElementById("totalSales").innerText="$"+total.toFixed(0)
document.getElementById("avgSales").innerText="$"+avg.toFixed(2)
document.getElementById("peakSales").innerText="$"+peak.toFixed(0)

Plotly.newPlot("salesChart",[{

x:dates,
y:sales,
fill:"tozeroy",
type:"scatter"

}])

}

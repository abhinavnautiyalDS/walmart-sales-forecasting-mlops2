const API_URL = "https://walmart-sales-forecasting-mlops2.onrender.com/predict"

let dataset = []

const upload = document.getElementById("csvUpload")
const runBtn = document.getElementById("runPrediction")

upload.addEventListener("change", async function(){

const file = upload.files[0]

if(!file){

alert("Please upload CSV")
return

}

const text = await file.text()

dataset = parseCSV(text)

alert("File loaded. Now click Run Prediction.")

})


runBtn.addEventListener("click", function(){

if(dataset.length === 0){

alert("Upload CSV first")
return

}

sendForPrediction(dataset)

})



function parseCSV(text){

const rows = text.trim().split("\n")
const headers = rows[0].split(",")

return rows.slice(1).map(row=>{

const values=row.split(",")

let obj={}

headers.forEach((h,i)=>{

obj[h.trim()] = values[i]

})

return obj

})

}


async function sendForPrediction(data){

try{

const res = await fetch(API_URL,{

method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify(data)

})

const result = await res.json()

renderCharts(result)

}catch(err){

console.error(err)

}

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
type:"scatter",
line:{color:"#0071dc"}

}])

}


document.getElementById("downloadTemplate").addEventListener("click", function(){

const csv = `Store,Date,Dept,IsHoliday,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment,Type,Size
1,2012-01-06,1,0,45,3.2,0,0,0,0,0,220,7.2,1,151315`

const blob = new Blob([csv], { type: 'text/csv' })

const link = document.createElement("a")

link.href = URL.createObjectURL(blob)

link.download = "walmart_template.csv"

link.click()

})


document.querySelectorAll(".tab").forEach(tab=>{

tab.addEventListener("click",()=>{

document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"))
document.querySelectorAll(".tab-content").forEach(c=>c.classList.remove("active"))

tab.classList.add("active")

const target=tab.dataset.tab

document.getElementById(target).classList.add("active")

})

})
